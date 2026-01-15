import os
import torch
import argparse
from parse_utils import *

import pickle
import numpy as np
import transformers
from transformers import EarlyStoppingCallback
from utils.prompter import Prompter
from model_t5 import T54Rec
from utils.data_utils import SequentialDataset, SequentialCollator, SequentialTestDataset
from utils.eval_utils import computeTopNAccuracy, print_results
import random

# === CF parallelization helper ===
from modules.cf_parallel import CFParallel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(args):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(args)

    print(torch.__version__)
    assert args.base_model, "Please specify a --base_model, e.g. --base_model='google-t5/t5-small'"
    set_seed(args.seed)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    print(gradient_accumulation_steps)
    prompter = Prompter(args.prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    # === Data ===
    dataset = SequentialDataset(args.data_path, 50, args.n_query, args.n_sem)

    # Load CF (SASRec) item embeddings; keep `user_embed` placeholder as before
    user_embed, item_embed = None, pickle.load(open(f'../data/{args.data_path}/SASRec_item_embed.pkl', 'rb'))

    prefix = args.data_path.split("/")[-2]
    feat_path = f"{args.data_path}{prefix}.emb-{args.sem_encoder}-tdcb.npy"
    item_feature = torch.FloatTensor(np.load(feat_path, allow_pickle=True))
    data_collator = SequentialCollator()

    # small eval dataset
    eval_dataset = SequentialTestDataset(args.data_path, 50, args.val_set_size, args.n_query, args.n_sem)

    # tokenizer args
    tokenizer_args = {
        "in_dim": item_feature.shape[-1],
        "layers": args.layers,
        "dropout_prob": args.dropout_prob,
        "bn": args.bn,
        "loss_type": args.loss_type,
        "item_feature": item_feature,
    }

    # === Model ===
    model = T54Rec(
        base_model=args.base_model,
        input_embeds=item_embed,          # keep original path: model builds embed layer from this
        cache_dir=args.cache_dir,
        device_map=device_map,
        input_dim=64,                     # CF input dim (SASRec vector size); unchanged
        instruction_text=prompter.generate_prompt(),
        user_embeds=user_embed,
        m_item=dataset.m_item,
        n_query=args.n_query,
        n_cf=args.n_cf,                   # now equals num_branches
        n_sem=args.n_sem,
        alpha=args.alpha,
        **tokenizer_args,
    )
    # model.t5_model.gradient_checkpointing_enable()

    # Attach CFParallel so the model can access K-way CF tokens if needed
    # Use the model's device to place the CF table
    model_device = next(model.parameters()).device
    # item_embed is typically a numpy array (n_items, cf_in_dim)
    # cf_mat_torch = torch.tensor(item_embed, dtype=torch.float32, device=model_device)
    cf_dtype = next(model.parameters()).dtype  # fp16 under fp16 training
    cf_mat_torch = torch.as_tensor(item_embed, dtype=cf_dtype, device=model_device)

    # Build CFParallel and attach
    model.cf_parallel = CFParallel(
        cf_table=cf_mat_torch,
        d_model=model.t5_model.config.d_model,   # T5 hidden size
        n_branches=args.num_branches,
        fuse_mode=args.branch_fuse
    ).to(model_device)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            gradient_checkpointing=False,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=50,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.val_set_size > 0 else None,
            save_steps=200,
            lr_scheduler_type=args.lr_scheduler,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=None,
            save_safetensors=False,
        ),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # === Evaluation ===
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.eval()
        topk = [5, 10]

        dataset = SequentialTestDataset(args.data_path, 50, args.val_set_size, args.n_query, args.n_sem)
        testData = dataset.testData

        gold_list = []
        pred_list = []
        warm_gold_list, warm_pred_list = [], []
        cold_gold_list, cold_pred_list = [], []

        # warm, cold items
        warm = list(np.load(args.data_path + "warm_item.npy", allow_pickle=True).tolist())
        cold = list(np.load(args.data_path + "cold_item.npy", allow_pickle=True).tolist())
        warm_tensor = torch.LongTensor(warm).to(model_device)
        cold_tensor = torch.LongTensor(cold).to(model_device)

        from tqdm import tqdm
        with torch.no_grad():
            best_recall = 0
            flag = 0
            for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] if model.n_sem else [0]:
                model.beta = torch.nn.Parameter(
                    torch.tensor([1 - beta] + [beta] * args.n_sem, dtype=torch.float, requires_grad=False)
                )
                print(f"beta: {beta} !!!")
                gold_list = []
                pred_list = []
                warm_gold_list, warm_pred_list = [], []
                cold_gold_list, cold_pred_list = [], []

                # === Build all_cf for ranking ===
                # original (n_cf=1): model.all_cf = model.input_proj(model.input_embeds[0](ids+1)).unsqueeze(0)
                idx_tensor = torch.arange(model.m_item, dtype=torch.long, device=model_device)

                if hasattr(model, "cf_parallel") and args.n_cf > 1:
                    # Use parallel CF heads to get shape (K, n_item, d_model)
                    # CFParallel.tokens expects (B,) → (B, K, d_model). Here B := n_item
                    all_ids = (idx_tensor + 1)  # consistent with original code (+1 for padding index 0)
                    cf_tok = model.cf_parallel.tokens(all_ids)          # (n_item, K, d_model)
                    model.all_cf = cf_tok.transpose(0, 1).contiguous()  # (K, n_item, d_model)
                else:
                    # Fallback to legacy single-branch path
                    model.all_cf = model.input_proj(model.input_embeds[0](idx_tensor + 1)).unsqueeze(0)  # (1, n_item, d_model)

                # Precompute semantics if present
                if model.n_sem:
                    model.recon_all = model.tokenize_all()  # (n_sem, n_item, dim)

                for u in tqdm(testData, total=len(testData)):
                    if len(testData[u]) == 0:
                        continue
                    inputs = torch.LongTensor(testData[u][0]).to(model_device).unsqueeze(0)  # seq, item
                    inputs_mask = torch.ones((inputs.shape[0], inputs.shape[1] * (args.n_query)), device=model_device)
                    _, ratings, _, _, _ = model.predict(inputs, inputs_mask, inference=True)  # ratings (bs, n_item)
                    groundTruth = testData[u][1]
                    # all
                    _, pred = torch.topk(ratings, k=topk[-1])
                    pred = pred.cpu().tolist()
                    gold_list.append(groundTruth)
                    pred_list.append(pred)
                    # warm and cold gold
                    warm_gold, cold_gold = [], []
                    for item in groundTruth:
                        if item in warm:
                            warm_gold.append(item)
                        else:
                            cold_gold.append(item)
                    warm_gold_list.append(warm_gold)
                    cold_gold_list.append(cold_gold)

                    # warm pred
                    import copy
                    warm_ratings = copy.deepcopy(ratings)
                    warm_ratings[..., cold_tensor] = -1e16
                    _, pred = torch.topk(warm_ratings, k=topk[-1])
                    pred = pred.cpu().tolist()
                    warm_pred_list.append(pred)

                    # cold pred
                    cold_ratings = copy.deepcopy(ratings)
                    cold_ratings[..., warm_tensor] = -1e16
                    _, pred = torch.topk(cold_ratings, k=topk[-1])
                    pred = pred.cpu().tolist()
                    cold_pred_list.append(pred)

                test_results = computeTopNAccuracy(gold_list, pred_list, topk)
                print("======= All performance")
                print_results(None, None, test_results)

                if test_results[1][0] > best_recall:
                    flag = 1
                    best_beta = beta
                    best_test_results = test_results
                    best_recall = test_results[1][0]

                test_results = computeTopNAccuracy(warm_gold_list, warm_pred_list, topk)
                print("======= Warm performance")
                print_results(None, None, test_results)

                if flag:
                    best_warm_results = test_results

                test_results = computeTopNAccuracy(cold_gold_list, cold_pred_list, topk)
                print("======= Cold performance")
                print_results(None, None, test_results)

                if flag:
                    best_cold_results = test_results
                    flag = 0

        print(f"=== End. Best beta is {best_beta}")
        print("======= All performance")
        print_results(None, None, best_test_results)
        print("======= Warm performance")
        print_results(None, None, best_warm_results)
        print("======= Cold performance")
        print_results(None, None, best_cold_results)

        model.t5_model.save_pretrained(args.output_dir)
        model.t5_tokenizer.save_pretrained(args.output_dir)
        model_path = os.path.join(args.output_dir, "adapter.pth")

        input_embeds, input_proj, tokenizer = (
            model.input_embeds.state_dict(),
            model.input_proj.state_dict(),
            model.tokenizer.state_dict(),
        )
        torch.save({"input_embeds": input_embeds, "input_proj": input_proj, "tokenizer": tokenizer}, model_path)

    del model


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="set_identifier")

    # New flags for CF parallelization
    parser.add_argument(
        "--num_branches", type=int, default=1,
        help="K: number of parallel CF branches (also equals n_cf)"
    )
    parser.add_argument(
        "--branch_fuse", type=str, default="mean",
        choices=["mean", "learned"],
        help="Fuse K branch logits by mean or learned gate"
    )
    parser.add_argument(
        "--cf_div", type=float, default=0.0,
        help="Diversity regularization weight between CF branches (training-time; optional)"
    )

    # Existing arg groups
    parser = train_args(parser)
    parser = identifier_args(parser)
    parser = wandb_args(parser)
    parser = parse_AE_args(parser)

    args = parser.parse_args()
    # Derive identifier width
    args.n_cf = args.num_branches
    args.n_query = args.n_sem + args.n_cf

    train(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from Q_t5 import *
from AE.models.ae import AE
from transformers import T5Tokenizer


def generate_position_ids(total_length, i, j, n, device=None):
    """
    :param total_length: seq_length
    :param i: instruction token length
    :param j: response token length
    :param n: n_query within each item
    :return: PyTorch tensor
    """
    assert i + j < total_length, "i + j should be smaller than total_length"
    # create base vector
    vector = torch.arange(total_length, dtype=torch.long, device=device)
    # mask middle item sequence part
    middle_mask = (vector >= i) & (vector < total_length - j)
    # change the middle item sequence part, let the position id be the same within the items
    vector[middle_mask] = i + torch.div((vector[middle_mask] - i), n, rounding_mode='floor')
    return vector


def new_compute_bias(self, seq_length, key_length, device=None, **kwargs):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device

    context_position = generate_position_ids(seq_length, self.n_instruct, self.n_response, self.n_query, device=device)[
                       :, None]
    memory_position = generate_position_ids(key_length, self.n_instruct, self.n_response, self.n_query, device=device)[
                      None, :]

    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values


def get_position_ids_for_qdecoder(self, seq_length, key_length, device=None, **kwargs):
    context_position = torch.zeros(seq_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.zeros(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values


class T54Rec(nn.Module):
    def __init__(self, **args):
        super(T54Rec, self).__init__()
        self.args = args
        self.n_query = args["n_query"]
        self.n_cf = args["n_cf"]
        self.n_sem = args["n_sem"]
        self.alpha = args["alpha"]  # coefficient of tokenizer loss


        print(f'Initializing language decoder ...')

        self.t5_model = QT5.from_pretrained(self.args['base_model'], local_files_only=False,
                                            cache_dir=args['cache_dir'], device_map=self.args['device_map'],
                                            n_query=self.args['n_query'])

        if args.get("inference", None) is None:
            nn.init.normal_(self.t5_model.decoder.query_emb.weight.data, mean=0,
                            std=0.02)  # manualy initialize, otherwise the initialization gives a very large range

        self.t5_model.config.use_cache = False

        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=False,
                                                        cache_dir=args['cache_dir'])
        self.t5_tokenizer.pad_token_id = 0
        self.t5_tokenizer.padding_side = "left"
        self.instruct_ids, self.instruct_mask = self.t5_tokenizer(self.args['instruction_text'][0],
                                                                  truncation=True, padding=False,
                                                                  return_tensors='pt',
                                                                  add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.t5_tokenizer(self.args['instruction_text'][1],
                                                                  truncation=True, padding=False,
                                                                  return_tensors='pt',
                                                                  add_special_tokens=False).values()
        self.input_dim = args['input_dim']
        print("instruction:", self.args['instruction_text'][0])
        print("response_split:", self.args['instruction_text'][1])

        self.replace_position_func()
        print('Language decoder initialized.')

        self.m_item = self.args['m_item']
        self.emb_dim = self.t5_model.config.hidden_size

        # initialize CF token
        padding_embed = torch.zeros(1, self.input_dim)
        concat_embed = torch.cat([padding_embed, self.args['input_embeds']], dim=0)
        self.input_embeds = nn.ModuleList([nn.Embedding.from_pretrained(concat_embed, freeze=True)])

        # initialize CF projector (cf token dim -> LLM token dim)
        self.input_proj = nn.Linear(self.input_dim, self.t5_model.config.hidden_size)

        # initialize sem tokenizer
        if self.n_sem:
            self.tokenizer = AE(in_dim=args["in_dim"],
                                e_dim=self.t5_model.config.hidden_size * (self.n_query - self.n_cf),
                                layers=args["layers"],
                                dropout_prob=args["dropout_prob"],
                                bn=args["bn"],
                                loss_type=args["loss_type"],
                                item_feature=args["item_feature"]
                                )

        # initialize CE Loss for recommendation
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing on inner T5. Older transformers versions do
        not accept kwargs (e.g. use_reentrant), so we swallow them gracefully.
        """
        if hasattr(self.t5_model, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs:
                    self.t5_model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
                else:
                    self.t5_model.gradient_checkpointing_enable()
            except TypeError:
                # Older transformers: method takes no kwargs
                self.t5_model.gradient_checkpointing_enable()
        if hasattr(self.t5_model, "config"):
            self.t5_model.config.use_cache = False  # required with checkpointing

    def gradient_checkpointing_disable(self):
        if hasattr(self.t5_model, "gradient_checkpointing_disable"):
            self.t5_model.gradient_checkpointing_disable()
        if hasattr(self.t5_model, "config"):
            self.t5_model.config.use_cache = True

    def replace_position_func(self):
        # replace compute_bias function
        for layer in self.t5_model.encoder.block:
            layer.layer[0].SelfAttention.n_query = self.n_query
            layer.layer[0].SelfAttention.n_instruct = self.instruct_ids.shape[-1]
            layer.layer[0].SelfAttention.n_response = self.response_ids.shape[-1]
            layer.layer[0].SelfAttention.compute_bias = new_compute_bias.__get__(layer.layer[0].SelfAttention)

        for layer in self.t5_model.decoder.block:
            layer.layer[0].SelfAttention.compute_bias = get_position_ids_for_qdecoder.__get__(
                layer.layer[0].SelfAttention)
            layer.layer[1].EncDecAttention.compute_bias = get_position_ids_for_qdecoder.__get__(
                layer.layer[1].EncDecAttention)

    def tokenize_all(self):
        # ensure AE and tensors use same device and we don't track grads
        device = next(self.parameters()).device
        self.tokenizer.to(device)  # <-- move AE params to GPU
        self.tokenizer.eval()  # <-- optional: inference mode for AE
        with torch.no_grad():  # <-- avoid unnecessary grads
            idx_tensor = torch.arange(self.m_item, dtype=torch.long, device=device)

            # keep item_feature on device too
            self.tokenizer.item_feature = self.tokenizer.item_feature.to(device)

            sem_input = self.tokenizer.item_feature[idx_tensor]  # (n_item, in_dim)
            sem_emb, recon_emb = self.tokenizer(sem_input)
            sem_emb = sem_emb.view(sem_emb.shape[0], self.n_sem, -1)
            self.all_sem = sem_emb.transpose(0, 1).contiguous().to(device)
            # self.all_sem = sem_emb.transpose(0, 1)
            return recon_emb

    def get_batch_sem(self, item_id, all_recon_emb):
        # device = next(self.parameters()).device

        '''
        This function takes the item id as input, and extract these items' semantic token embeddings from all sem token embeddings.
        '''
        # initialize a zero tensor
        # item_sem_token = torch.zeros(item_id.shape[0], item_id.shape[1], self.n_sem, self.emb_dim,
        #                              dtype=self.all_sem.dtype).cuda()  # (bs, seq_len, n_sem, dim)

        item_sem_token = torch.zeros(item_id.shape[0], item_id.shape[1], self.n_sem, self.emb_dim,
                                     dtype=self.all_sem.dtype, device=item_id.device)

        non_zero_mask = self.batch_nonzero_mask  # (bs, seq_len)
        non_zero_ids = item_id[non_zero_mask]  # (n_nonzero, )

        # item_sem_token[non_zero_mask] = self.all_sem.transpose(0, 1)[non_zero_ids - 1]  # (n_nonzero, n_sem, dim)
        bank = self.all_sem.to(item_id.device).transpose(0, 1)  # <— ensure same device
        item_sem_token[non_zero_mask] = bank[non_zero_ids - 1]

        sem_input = self.tokenizer.item_feature.to(item_id.device)[non_zero_ids - 1]  # <— same
        recon_emb = all_recon_emb.to(item_id.device)[non_zero_ids - 1]  # <— same
        sem_emb = bank[non_zero_ids - 1]
        return sem_input, sem_emb, recon_emb, item_sem_token

    def _build_batch_cf_tokens(self, inputs_id):
        """
        Build CF tokens for a batch of sequences.
        - If self.cf_parallel exists and n_cf > 1, produce (bs, seq_len, K, dim) using K parallel heads.
        - Else, fall back to legacy single CF token: (bs, seq_len, 1, dim).
        """
        bs, seq_len = inputs_id.shape[0], inputs_id.shape[1]
        device = inputs_id.device

        if hasattr(self, "cf_parallel") and self.n_cf > 1:
            # Flatten ids and handle padding (id==0). CFParallel expects 0-based item ids.
            flat_ids = inputs_id.view(-1)  # (bs*seq_len,)
            non_zero = flat_ids.nonzero(as_tuple=False).squeeze(-1)  # indices where id != 0
            cf_tok = torch.zeros(bs * seq_len, self.n_cf, self.emb_dim, device=device)

            if non_zero.numel() > 0:
                ids0 = flat_ids[non_zero] - 1  # 0-based
                # CFParallel.tokens: (N_nonzero, K, dim)
                cf_nonzero = self.cf_parallel.tokens(ids0)  # (Nnz, K, emb_dim)
                cf_tok[non_zero] = cf_nonzero

            cf_tok = cf_tok.view(bs, seq_len, self.n_cf, self.emb_dim)  # (bs, seq_len, K, dim)
            return cf_tok
        else:
            # Legacy: one CF token
            cf_vec = self.input_embeds[0](inputs_id)  # (bs, seq_len, input_dim)
            cf_tok = self.input_proj(cf_vec).unsqueeze(2)  # (bs, seq_len, 1, dim)
            return cf_tok

    def _expand_beta_if_needed(self, beta):
        """
        Expand legacy beta ([beta_cf, beta_sem, beta_sem, ...] with length 1+n_sem)
        to new length (n_cf + n_sem) by splitting CF mass evenly across K=n_cf.
        """
        if beta.numel() == (1 + self.n_sem) and self.n_cf > 1:
            device = beta.device
            cf_w = beta[0].item()
            sem_w = beta[1:].clone().detach()
            cf_vec = torch.full((self.n_cf,), fill_value=cf_w / float(self.n_cf), device=device, dtype=beta.dtype)
            new_beta = torch.cat([cf_vec, sem_w], dim=0)  # (n_cf + n_sem,)
            return new_beta
        return beta

    def predict(self, inputs_id, inputs_mask, inference=False):
        device = next(self.parameters()).device

        self.batch_nonzero_mask = inputs_id != 0

        if not inference and self.n_sem:
            self.recon_all = self.tokenize_all()

        x, unique_sem_emb, x_hat, item_sem_token = None, None, None, None

        # 1. get semantic tokens
        if self.n_sem:
            # put current batch item into the AE to get the sem emb
            x, unique_sem_emb, x_hat, item_sem_token = self.get_batch_sem(inputs_id, self.recon_all)
            # x: (n_nonzero, input_dim)
            # unique_sem_emb (n_nonzero, n_sem, dim)
            # x_hat (n_nonzero, input_dim)
            # item_sem_token (bs, seq_len, n_sem, dim)

        bs = inputs_id.shape[0]
        # instruct_embeds = self.t5_model.shared(self.instruct_ids.cuda()).expand(bs, -1, -1)
        # response_embeds = self.t5_model.shared(self.response_ids.cuda()).expand(bs, -1, -1)
        # answer_embeds = self.t5_model.decoder.query_emb(self.t5_model.decoder.query_input_ids.cuda()).expand(bs, -1, -1)
        #
        # instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        # response_mask = self.response_mask.cuda().expand(bs, -1)

        # device = inputs_id.device
        # instruct_embeds = self.t5_model.shared(self.instruct_ids.to(device)).expand(bs, -1, -1)
        # ids = self.instruct_ids.to(self.t5_model.shared.weight.device)
        # Figure out the real devices from the embedding weights
        emb_device = self.t5_model.shared.weight.device
        q_device = self.t5_model.decoder.query_emb.weight.device  # decoder-side query embedding

        # Build fixed-geometry token embeddings on the right devices
        instruct_embeds = self.t5_model.shared(self.instruct_ids.to(emb_device)).expand(bs, -1, -1)
        response_embeds = self.t5_model.shared(self.response_ids.to(emb_device)).expand(bs, -1, -1)
        answer_embeds = self.t5_model.decoder.query_emb(
            self.t5_model.decoder.query_input_ids.to(q_device)
        ).expand(bs, -1, -1)

        # And the masks too
        instruct_mask = self.instruct_mask.to(emb_device).expand(bs, -1)
        response_mask = self.response_mask.to(emb_device).expand(bs, -1)

        # 2. get cf tokens (K-way if enabled)
        cf_tokens = self._build_batch_cf_tokens(inputs_id)  # (bs, seq_len, n_cf, dim) with n_cf>=1

        # 3. concat cf emb and sem emb
        if self.n_sem:
            inputs = torch.cat([cf_tokens, item_sem_token], dim=2)  # (bs, seq_len, n_cf+n_sem, dim)
        else:
            inputs = cf_tokens  # (bs, seq_len, n_cf, dim)
        inputs = inputs.view(bs, -1, self.t5_model.config.hidden_size)

        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        # 4. query-guided simultaneous decoding (t5 forward)
        outputs = self.t5_model(inputs_embeds=inputs, attention_mask=attention_mask,
                                decoder_inputs_embeds=answer_embeds, return_dict=True, output_hidden_states=True)

        # 5. grounding
        query_output = outputs.decoder_hidden_states[-1][:, -self.n_query:]  # (bs, n_query, dim)
        output_emb = query_output.transpose(0, 1)  # (n_query, bs, emb_dim)

        if not inference:
            # get all cf (K-way if enabled) and all sem
            if hasattr(self, "cf_parallel") and self.n_cf > 1:
                # idx_tensor = torch.arange(self.m_item, dtype=torch.long).cuda()  # 0..m_item-1
                device = inputs_id.device
                idx_tensor = torch.arange(self.m_item, dtype=torch.long, device=device)

                # CFParallel.tokens expects 0-based ids
                cf_all = self.cf_parallel.tokens(idx_tensor)  # (n_item, K, dim)
                self.all_cf = cf_all.transpose(0, 1).contiguous()  # (K, n_item, dim)
            else:
                # idx_tensor = torch.arange(self.m_item, dtype=torch.long).cuda()
                device = inputs_id.device
                idx_tensor = torch.arange(self.m_item, dtype=torch.long, device=device)

                self.all_cf = self.input_proj(self.input_embeds[0](idx_tensor + 1)).unsqueeze(0)  # (1, n_item, dim)

            # in T54Rec.predict(), right before the concat
            if self.n_sem:
                dev = self.all_cf.device
                self.all_sem = self.all_sem.to(dev)
            mat = torch.cat([self.all_cf, self.all_sem], dim=0) if self.n_sem else self.all_cf

            # concat cf and sem, then dot product
            # mat = torch.cat([self.all_cf, self.all_sem], dim=0) if self.n_sem else self.all_cf  # (n_query, n_item, dim)
            if self.n_sem:
                self.all_sem = self.all_sem.to(self.all_cf.device)
                mat = torch.cat([self.all_cf, self.all_sem], dim=0)  # (n_query, n_item, dim)
            else:
                mat = self.all_cf

            mat = mat.transpose(1, 2)  # (n_query, dim, n_item)
            output = torch.bmm(output_emb, mat)  # (n_query, bs, n_item)
            return output_emb, outputs, output, x, unique_sem_emb, x_hat

        if inference:
            # concat cf and sem, then dot product
            # mat = torch.cat([self.all_cf, self.all_sem], dim=0) if self.n_sem else self.all_cf  # (n_query, n_item, dim)
            # make sem live on the same device as cf (belt-and-suspenders)
            if self.n_sem:
                self.all_sem = self.all_sem.to(self.all_cf.device)
                mat = torch.cat([self.all_cf, self.all_sem], dim=0)  # (n_query, n_item, dim)
            else:
                mat = self.all_cf

            mat = mat.transpose(1, 2)  # (n_query, dim, n_item)
            output = torch.bmm(output_emb, mat)  # (n_query, bs, n_item)

            # Expand legacy beta if needed (when it was created for n_cf==1)
            beta_vec = self._expand_beta_if_needed(self.beta)
            weight = beta_vec[:, None, None].to(self.t5_model.device)  # (n_query, 1, 1)
            score = torch.sum(output * weight, dim=0).squeeze()
            return outputs, score, x, unique_sem_emb, x_hat

    def forward(self, inputs, inputs_mask, labels):
        _, outputs, scores, x, _, x_hat = self.predict(inputs, inputs_mask)  # prediction (bs, n_query, dim)

        loss = None
        logits = None
        if labels is not None:
            # generative recommendation loss
            scores = torch.cat([_ for _ in scores], dim=0)
            all_item_gold = labels.squeeze().repeat(1, self.n_query).squeeze()  # (n_query*bs)
            loss = self.criterion(scores, all_item_gold)


            # tokenizer loss (i.e., autoencoder loss)
            loss_ae = 0
            if self.n_sem:
                loss_ae = F.mse_loss(x, x_hat, reduction='mean')

            # Eq.(8) in paper
            loss += self.alpha * loss_ae

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

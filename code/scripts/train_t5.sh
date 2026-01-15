#!/usr/bin/env bash
set -euo pipefail

dataset=${1}          # toys|beauty|sports|steam
lr=${2}               # e.g. 1e-3
n_sem=${3}            # e.g. 4
alpha=${4}            # e.g. 0.7

# Optional CF-parallelization args (defaults if omitted)
num_branches="${5:-1}"     # K branches
branch_fuse="${6:-mean}"   # mean or learned
cf_div="${7:-0.0}"         # diversity reg weight

# Optional: explicit checkpoint path (8th arg). If omitted, we auto-pick latest.
explicit_ckpt="${8-}"

seed=42
AE_layers="512 256 128"

# Derived (for naming only; finetune_t5.py recomputes internally)
n_cf="${num_branches}"
n_query=$(( n_sem + n_cf ))

# Paths
outdir="../output/"
logdir="../log/t5/${dataset}"
mkdir -p "${logdir}"

# Resume logic (safe under set -u)
resume_opt=()
if [[ -n "${explicit_ckpt}" ]]; then
  echo ">> Resuming from checkpoint (explicit): ${explicit_ckpt}"
  resume_opt=(--resume_from_checkpoint "${explicit_ckpt}")
else
  latest_ckpt="$(ls -td "${outdir}"/checkpoint-* 2>/dev/null | head -1 || true)"
  if [[ -n "${latest_ckpt}" ]]; then
    echo ">> Resuming from checkpoint (auto): ${latest_ckpt}"
    resume_opt=(--resume_from_checkpoint "${latest_ckpt}")
  else
    echo ">> No checkpoint found. Starting fresh."
  fi
fi

# GPU + allocator hints
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

# Launch (2 GPUs for validation; scale up later)
torchrun --nproc_per_node=2 --master_port=12522 finetune_t5.py \
  --base_model t5-small \
  --data_path ../data/toys/ \
  --cache_dir ../cache/ \
  --output_dir "${outdir}" \
  --seed ${seed} \
  --batch_size 512 \
  --micro_batch_size 64 \
  --num_epochs 20 \
  --learning_rate 1e-3 \
  --cutoff_len 256 \
  --val_set_size 2000 \
  --lr_scheduler cosine \
  --warmup_steps 100 \
  --n_sem ${n_sem} \
  --num_branches ${num_branches} \
  --branch_fuse ${branch_fuse} \
  --cf_div ${cf_div} \
  "${resume_opt[@]}"

# examples:
# bash scripts/train_t5.sh toys 1e-3 4 0.7
# bash scripts/train_t5.sh toys 1e-3 4 0.7 2 mean
# bash scripts/train_t5.sh toys 8e-4 4 0.7 3 learned 1e-3
# bash scripts/train_t5.sh toys 1e-3 4 0.7 1 mean 0.0 ../output/checkpoint-2200

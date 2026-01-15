dataset=${1}
n_sem=${2}
ckpt=${3}
n_query=$((n_sem + 1))
n_cf=1

python inference_t5.py \
    --base_model /your/path/to/t5 \
    --ckpt_dir ${ckpt} \
    --sem_encoder t5 \
    --data_path ../data/${dataset}/ \
    --out_proj True \
    --n_query ${n_query} \
    --n_sem ${n_sem} \
    --n_cf ${n_cf} \
    --cutoff_len 512 \

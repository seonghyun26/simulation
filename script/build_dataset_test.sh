cd ../util

CUDA_VISIBLE_DEVICES=7 python build_dataset.py \
    --molecule alanine \
    --state alpha_P \
    --temperature 100
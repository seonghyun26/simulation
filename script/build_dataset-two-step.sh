cd ../util

CUDA_VISIBLE_DEVICES=$1 python build_dataset.py \
    --molecule alanine \
    --state c5 \
    --temperature 500.0 \
    --index two-step \
    --data_size 1000000 \
    --dataset_index regression-v1
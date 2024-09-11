cd ../util

CUDA_VISIBLE_DEVICES=$1 python build_dataset.py \
    --molecule alanine \
    --state c5 \
    --temperature 500.0 \
    --index multi-next \
    --data_size 100000 \
    --dataset_index v2
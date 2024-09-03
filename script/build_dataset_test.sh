cd ../util

CUDA_VISIBLE_DEVICES=$1 python build_dataset.py \
    --molecule alanine \
    --state c5 \
    --temperature 273.0 \
    --index random 
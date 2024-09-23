cd ../

python build_dataset.py \
    --molecule alanine \
    --state c5 \
    --temperature 300.0 \
    --dataset_size 100000 \
    --dataset_type multi-next \
    --dataset_version v3 \
    --sim_repeat_num 16
cd ../

python build_dataset.py \
    --molecule alanine \
    --state c5 \
    --temperature 500.0 \
    --sim_length 10000000 \
    --dataset_type random \
    --dataset_version v2
cd ../

python build_dataset-multi-temp.py \
    --molecule alanine \
    --state c5 \
    --temperature 300.0 \
    --sim_length 1000000 \
    --dataset_type multi-temp \
    --dataset_version v3
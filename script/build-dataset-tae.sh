cd ../


python build_tae_dataset.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 10000 \
    --dataset_version tae-10n-v1 \
    --negative_sample_augmentation 3000 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 
cd ../

python build_cl_dataset_v2.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 1000000 \
    --dataset_version 10n-v2 \
    --positive_sample_augmentation 100 \
    --negative_sample_augmentation 10000 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 
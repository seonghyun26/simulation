cd ../

python build_cl_dataset_v2.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 100000 \
    --dataset_version v8 \
    --positive_sample_augmentation 10 \
    --negative_sample_augmentation 100 \
    --traj_dir 24-12-26/14:53 \
    --traj_dir 24-12-26/15:08 
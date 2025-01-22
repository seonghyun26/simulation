cd ../

# 10ns 2 trajectories each
python build_cl_dataset_v2.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 100000 \
    --dataset_version 10n-v5 \
    --positive_sample_augmentation 10 \
    --negative_sample_augmentation 10000 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 


# 250ns 5 trajectories each
# python build_cl_dataset_v2.py \
#     --molecule alanine \
#     --temperature 300.0 \
#     --dataset_size 10000 \
#     --dataset_version 250n-v3 \
#     --positive_sample_augmentation 1 \
#     --negative_sample_augmentation 1000 \
#     --traj_dir 25-01-17/01:51 \
#     --traj_dir 25-01-17/10:06 
cd ../

python build_da_dataset.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 10000 \
    --dataset_version da-1n-v1 \
    --traj_dir 24-12-26/14:53 \
    --traj_dir 24-12-26/15:08

python build_da_dataset.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 10000 \
    --dataset_version da-10n-v1 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 

python build_da_dataset.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 10000 \
    --dataset_version da-250n-v1 \
    --traj_dir 25-01-17/01:51 \
    --traj_dir 25-01-17/10:06 

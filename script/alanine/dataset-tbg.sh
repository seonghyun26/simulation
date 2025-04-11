cd ../../


CUDA_VISIBLE_DEVICES=$1 python dataset-tbg.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 10000 \
    --dataset_version tbg-10n-v2 \
    --time_lag 30 \
    --traj_dir 10nano/c5-0 \
    --traj_dir 10nano/c5-1 \
    --traj_dir 10nano/c5-2 \
    --traj_dir 10nano/c5-3 \
    --traj_dir 10nano/c5-4 \
    --traj_dir 10nano/c7ax-0 \
    --traj_dir 10nano/c7ax-1 \
    --traj_dir 10nano/c7ax-2 \
    --traj_dir 10nano/c7ax-3 \
    --traj_dir 10nano/c7ax-4




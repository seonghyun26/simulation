cd ../../

python dataset-random.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 1000 \
    --dataset_version 1n-v1 \
    --traj_dir 1nano/c5-0 \
    --traj_dir 1nano/c5-1 \
    --traj_dir 1nano/c5-2 \
    --traj_dir 1nano/c5-3 \
    --traj_dir 1nano/c5-4 \
    --traj_dir 1nano/c7ax-0 \
    --traj_dir 1nano/c7ax-1 \
    --traj_dir 1nano/c7ax-2 \
    --traj_dir 1nano/c7ax-3 \
    --traj_dir 1nano/c7ax-4


python dataset-random.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 1000 \
    --dataset_version 10n-v1 \
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
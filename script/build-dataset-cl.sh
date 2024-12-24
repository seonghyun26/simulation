cd ../

python build_cl_dataset.py \
    --molecule alanine \
    --temperature 1200.0 \
    --dataset_size 100000 \
    --dataset_version v4 \
    --traj_dir 24-11-14/14:08 \
    --traj_dir 24-11-14/14:20 
    # --traj_dir 24-11-13/16:32 \
    # --traj_dir 24-11-13/16:19 \
    # --traj_dir 24-11-11/17:13 \
    # --traj_dir 24-11-12/10:58 
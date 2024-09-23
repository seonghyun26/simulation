CUDA_VISIBLE_DEVICES=$1 python sim.py\
    --device cuda\
    --num_steps 10000 \
    --timestep 0.01 \
    --num_samples 1024 \
    --temperature $2
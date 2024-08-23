cd ../../

CUDA_VISIBLE_DEVICES=4 python main.py \
    --molecule alanine \
    --state c5 \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 400 \
    --time 100_000_000 \
    --platform OpenCL \
    --precision mixed &

sleep 10

CUDA_VISIBLE_DEVICES=5 python main.py \
    --molecule alanine \
    --state c5 \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 500 \
    --time 100_000_000 \
    --platform OpenCL \
    --precision mixed &

sleep 10

CUDA_VISIBLE_DEVICES=6 python main.py \
    --molecule alanine \
    --state c7ax \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 400 \
    --time 100_000_000 \
    --platform OpenCL \
    --precision mixed &

sleep 10

CUDA_VISIBLE_DEVICES=7 python main.py \
    --molecule alanine \
    --state c7ax \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 500 \
    --time 100_000_000 \
    --platform OpenCL \
    --precision mixed &

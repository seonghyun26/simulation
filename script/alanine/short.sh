cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule alanine \
  --state $2 \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 273 \
  --time 100_000 \
  --platform OpenCL \
  --precision mixed \


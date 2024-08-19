cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule ad-c5 \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 600 \
  --time 100_000_000 \
  --platform OpenCL \
  --precision mixed

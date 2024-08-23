cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule alanine \
  --state c7ax \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 300 \
  --time 100_000_000 \
  --platform OpenCL \
  --precision mixed

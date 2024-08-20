cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule alanine \
  --state $2 \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 200 \
  --time 100 \
  --platform OpenCL \
  --precision mixed \
  --freq_csv 1


cd ../../


CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule alanine \
  --state $2 \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 300 \
  --time 250_000_000 \
  --platform OpenCL \
  --precision mixed \
  --log_stdout True \
  --freq_stdout 100000 \
  --log_dcd True \
  --freq_dcd 100 \
  --log_csv True \
  --freq_csv 100 \
  --log_force False

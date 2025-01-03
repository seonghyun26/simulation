cd ../../

for state in "c5" "c7ax"
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --molecule alanine \
        --state $state \
        --force_field amber99 \
        --solvent tip3p \
        --temperature 300 \
        --time 10_000_000 \
        --platform OpenCL \
        --precision mixed \
        --log_stdout True \
        --freq_stdout 10_000
done

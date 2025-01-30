cd ../../

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$((i+2)) python main.py \
        --molecule alanine \
        --state c7ax \
        --force_field amber99 \
        --solvent tip3p \
        --temperature 300 \
        --time 1_000_000 \
        --platform OpenCL \
        --precision mixed \
        --index $i &
    sleep 1
done


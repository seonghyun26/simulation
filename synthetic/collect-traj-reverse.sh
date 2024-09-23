for i in {1..4}
do
    temp=$((300 * i))
    CUDA_VISIBLE_DEVICES=$(($1+i)) python dynamics.py\
        --device cuda\
        --reverse True \
        --num_steps 10000 \
        --temperature $temp &
    sleep 1
done
for i in {1..4}
do
    temp=$((300 * i))
    CUDA_VISIBLE_DEVICES=$(($1+i)) python dynamics.py\
        --device cuda\
        --temperature $temp &
    sleep 1
done
cd ../

# State list: alpha_P, c5, c7ax
TEMP=( 100.0 200.0 400.0 500.0 )
for (( i=0; i<${#TEMP[@]}; i++ ));
do
    echo Building dataset for temperature ${TEMP[i]} K
    CUDA_VISIBLE_DEVICES=$(($i + 3)) python build_dataset.py \
        --molecule alanine \
        --state $1 \
        --temperature ${TEMP[i]} \
        --index random &
    sleep 1
done
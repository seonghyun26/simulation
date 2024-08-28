cd ../util

# State list: alpha_P, c5, c7ax
TEMP=( 100.0 200.0 300.0 400.0 500.0 600.0 )
for (( i=0; i<${#TEMP[@]}; i++ ));
do
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python build_dataset.py \
        --molecule alanine \
        --state $1 \
        --temperature ${TEMP[i]} \
        --index random &
done
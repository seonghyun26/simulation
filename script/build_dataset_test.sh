cd ../util

TEMP=( 100.0 200.0 300.0 400.0 500.0 600.0 )
for (( i=0; i<${#TEMP[@]}; i++ ));
do
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python build_dataset.py \
        --molecule alanine \
        --state alpha_P \
        --temperature ${TEMP[i]} &
    
    sleep 1
    
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python build_dataset.py \
        --molecule alanine \
        --state c5 \
        --temperature ${TEMP[i]} &

    sleep 1
    
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python build_dataset.py \
        --molecule alanine \
        --state c7ax \
        --temperature ${TEMP[i]} &
done
cd ../util

python build_dataset.py \
    --molecule alanine \
    --state c7ax \
    --temperature 300.0 \
    --index goal \
    --data_size 10000000
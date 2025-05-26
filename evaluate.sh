#!bin/bash

data_file="$1"
model_file="$2"
if [ ! -f "$data_file" ]; then 
    echo "data file: \"$data_file\" does not exist"
    echo "usage: predict.sh [path/to/data_file.csv]"
    exit 1
fi

if [ ! -f "$model_file" ]; then 
    echo "model file: \"$model_file\" does not exist"
    echo "usage: predict.sh [path/to/model_file.csv]"
    exit 1
fi

./output/evaluate.out $data_file $model_file > "./output/evaluate.txt"
cat "./output/evaluate.txt"
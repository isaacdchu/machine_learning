#!bin/bash

data_file="$1"
model_file="$2"
contains_label="$3"
output_file=${4:-"./output/predict.txt"}
if [ ! -f "$data_file" ]; then 
    echo "data file: \"$data_file\" does not exist"
    echo "usage: predict.sh [path/to/data_file.csv] [true|false]"
    exit 1
fi

if [ ! -f "$model_file" ]; then 
    echo "model file: \"$model_file\" does not exist"
    echo "usage: predict.sh [path/to/model_file.csv] [true|false]"
    exit 1
fi

if [[ "$contains_label" != "true" && "$contains_label" != "false" ]]; then
    echo "contains_label must be either 'true' or 'false'"
    echo "usage: predict.sh [path/to/data_file.csv] [path/to/model_file.csv] [true|false]"
    exit 1
fi

./output/predict.out $data_file $model_file $contains_label> "$output_file"
cat "$output_file"
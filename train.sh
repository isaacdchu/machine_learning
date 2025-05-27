#!bin/bash

data_file="$1"
params_file="$2"
model_path=${4:-"./output/logistic_regression.model"}
if [ ! -f "$data_file" ]; then 
    echo "data file:\"$data_file\" does not exist"
    echo "usage: train.sh path/to/data_file.csv path/to/params_file.csv [path/to/save/model]"
    exit 1
fi

if [ ! -f "$params_file" ]; then 
    echo "data file: \"$params_file\" does not exist"
    echo "usage: train.sh path/to/data_file.csv path/to/params_file.csv [path/to/save/model]"
    exit 1
fi

./output/train.out $data_file $params_file $model_path
cat $params_file
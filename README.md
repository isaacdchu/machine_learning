### RUN INSTRUCTIONS:
- First, ensure that data is formatted correctly (CSV file described below)
- Then, you can run:
```bash
make compile
bash train.sh ./data/<name>.csv ./conf/params.txt
bash evaluate.sh ./data/<name>.csv ./conf/params.txt
bash predict.sh ./data/<name>.csv ./conf/params.txt <contains_label> [<output_file>]
```

### DATA FORMATTING:
- Data must take the form of a CSV file with these extra conditions
    - Header titles:
        - Titles themselves may not include commas
        - Titles cannot be empty
        - There must be the same number of header titles as data columns
    - Label column:
        - One label column may be provided, and it must be the last column
    - Data format:
        - All data must be numerical in decimal representation (for example, 3.14, -3, 07, or 0.0)
        - No missing data / NaN / null values

### USE IN OTHER C++ PROGRAMS:
- Only files from `./src/Model/` need to be copied
- Then, you can use any class derived from `model`
    - Note that you must instantiate the models with parameters or model info

data from
https://www.kaggle.com/datasets/dragonheir/logistic-regression/data
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data

TODO:
- documentation
- Params and Info classes
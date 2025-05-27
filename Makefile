.PHONY: compile model_train model_predict model_evaluate clean

# Targets
compile: output/train.out output/evaluate.out output/predict.out
model_train: output/train.out
	@bash train.sh ./data/train_loan_data.csv ./conf/params.txt

model_evaluate: output/evaluate.out
	@bash evaluate.sh ./data/evaluate_loan_data.csv ./output/logistic_regression.model

model_predict: output/predict.out
	@bash predict.sh ./data/evaluate_loan_data.csv ./output/logistic_regression.model true

# Build outputs
output/train.out: src/Train/train.cpp src/Model/logistic_regression.cpp \
    src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp \
    src/Utils/utils.h src/Utils/math_utils.h src/Utils/data_utils.h src/Model/model.h src/Model/logistic_regression.h src/Train/train.h
	clang++ src/Train/train.cpp src/Model/logistic_regression.cpp src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp -o output/train.out -Isrc/Train -Isrc/Utils -Isrc/Model \
	-std=c++23

output/evaluate.out: src/Evaluate/evaluate.cpp src/Model/logistic_regression.cpp \
    src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp \
    src/Utils/utils.h src/Utils/math_utils.h src/Utils/data_utils.h src/Model/model.h src/Model/logistic_regression.h src/Evaluate/evaluate.h
	clang++ src/Evaluate/evaluate.cpp src/Model/logistic_regression.cpp src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp -o output/evaluate.out -Isrc/Evaluate -Isrc/Utils -Isrc/Model \
	-std=c++23

output/predict.out: src/Predict/predict.cpp src/Model/logistic_regression.cpp \
    src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp \
    src/Utils/utils.h src/Utils/math_utils.h src/Utils/data_utils.h src/Model/model.h src/Model/logistic_regression.h src/Predict/predict.h
	clang++ src/Predict/predict.cpp src/Model/logistic_regression.cpp src/Utils/utils.cpp src/Utils/math_utils.cpp src/Utils/data_utils.cpp -o output/predict.out -Isrc/Predict -Isrc/Utils -Isrc/Model \
	-std=c++23

# Clean
clean:
	rm -f output/*.out

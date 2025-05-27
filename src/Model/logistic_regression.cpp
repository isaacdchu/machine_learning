#include "model.h"
#include "../Utils/utils.h"
#include "../Utils/math_utils.h"
#include "../Utils/data_utils.h"
#include "logistic_regression.h"

LogisticRegression::LogisticRegression(const std::string &params_path) {
    handle_params(params_path);
}

void LogisticRegression::load_data(const std::string &data_path, bool contains_label){
    this->data_path = data_path;
    this->contains_label = contains_label;
    outliers = get_outliers(data_path, contains_label, params.outlier_std);
    info.min = get_min(data_path, contains_label, outliers);
    info.max = get_max(data_path, contains_label, outliers);
    if (!empty_model) {
        return;
    }
    int num_features = get_num_features(data_path, contains_label);
    info.weights.resize(num_features);
    for (int i = 0; i < num_features; ++i){
        info.weights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f; // small random values
    }
    info.bias = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f;

    // Prepare validation data for early stopping
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        std::cout << "(load_data) Unable to open file" << std::endl;
        throw 3;
    }
    
    std::ofstream tmp_file(tmp_path, std::ios::trunc);
    if (!tmp_file.is_open()){
        std::cout << "(load_data) Unable to open tmp file" << std::endl;
        throw 3;
    }
    // Read data file and write every 5th line to tmp file
    std::string line;
    getline(data_file, line); // Skip header line
    int line_num = 0;
    int i_outlier = -1;
    while (getline(data_file, line)) {
        if (outliers.find(++i_outlier) != outliers.end()) {
            // Skip outliers
            continue;
        }
        if (++line_num % 5 == 0){
            // Print line to "./tmp/tmp_train.txt" for early stopping
            tmp_file << line << std::endl;
        }
    }
    tmp_file.close();
    data_file.close();
}

void LogisticRegression::save_model(const std::string &model_path) {
    assert(!data_path.empty() && "Saving model requires data path");
    std::ofstream model_file(model_path);
    if (!model_file.is_open()) {
        std::cout << "Unable to open model file" << std::endl;
        throw 3;
    }

    // Save parameters
    model_file << params.learning_rate << std::endl;
    model_file << params.batch_size << std::endl;
    model_file << params.num_epochs << std::endl;
    model_file << params.threshold << std::endl;
    model_file << params.outlier_std << std::endl;
    model_file << params.early_stopping_threshold << std::endl;

    // Save model info
    print_vector(info.weights, model_file);
    model_file << info.bias << std::endl;
    print_vector(info.min, model_file);
    print_vector(info.max, model_file);

    model_file.close();
}

void LogisticRegression::print_model() {
    // Print model params
    std::cout << params.learning_rate << std::endl;
    std::cout << params.batch_size << std::endl;
    std::cout << params.num_epochs << std::endl;
    std::cout << params.threshold << std::endl;
    std::cout << params.outlier_std << std::endl;
    std::cout << params.early_stopping_threshold << std::endl;

    // Print model info
    print_vector(info.weights);
    std::cout << info.bias << std::endl;
    print_vector(info.min);
    print_vector(info.max);
}

void LogisticRegression::train(){
    assert(!data_path.empty()&& "Training requires data path");
    assert(contains_label && "Training requires labeled data");
    std::ifstream data_file(data_path);
        if (!data_file.is_open()) {
            std::cout << "(train) Unable to open file" << std::endl;
            throw 3;
        }
    std::string line;
    float prev_val_loss = MAXFLOAT;
    for (int epoch = 0; epoch < params.num_epochs; ++epoch) {
        std::cout<< "Epoch: " << epoch + 1 << "/" << params.num_epochs << std::endl;
        std::vector<float> label_value_batch(params.batch_size);
        std::vector<float> prediction_batch(params.batch_size);
        getline(data_file, line); // Skip header line
        int i_batch = 0;
        int line_num = 0;
        int i_outlier = 0;
        while (getline(data_file, line)) {
            if (outliers.find(i_outlier) != outliers.end()) {
                i_outlier++;
                // Skip outliers
                continue;
            }
            std::vector<float> processed_line = process_line(parse_csv_line(&line));
            std::vector<float> feature_values = normalize_data(get_feature_values(processed_line), info.min, info.max);
            label_value_batch[i_batch] = get_label_value(processed_line);
            prediction_batch[i_batch] = make_prediction(feature_values);
            i_batch++;
            if (i_batch == params.batch_size) {
                // Adjust weights and bias for the batch
                std::vector<float> errors = loss_gradient(prediction_batch, label_value_batch);
                for (int i = 0; i < params.batch_size; ++i) {
                    for (size_t j = 0; j < info.weights.size(); ++j) {
                        info.weights[j] -= params.learning_rate * errors[i] * feature_values[j];
                    }
                    info.bias -= params.learning_rate * errors[i];
                }
                i_batch = 0; // Reset batch index
            }
        }
        if (epoch % 5 != 0) {
            data_file.clear();
            data_file.seekg(0, std::ios::beg);
            continue;
        }
        // Early stopping
        std::string tmp_path = "./tmp/tmp_train.txt";
        std::ifstream val_file(tmp_path);
        if (!val_file.is_open()) {
            std::cout << "(train) Unable to open tmp file for early stopping" << std::endl;
            throw 3;
        }
        float val_loss = 0.0f;
        while (getline(val_file, line)) {
            std::vector<float> processed_line = process_line(parse_csv_line(&line));
            std::vector<float> feature_values = normalize_data(get_feature_values(processed_line), info.min, info.max);
            float label_value = get_label_value(processed_line);
            float prediction = make_prediction(feature_values);
            val_loss += loss(prediction, label_value);
        }
        val_file.close();
        if (val_loss > prev_val_loss * (1.0f + params.early_stopping_threshold)) {
            std::cout << "Early stopping at epoch " << epoch << " with validation loss: " << val_loss << std::endl;
            break;
        }
        prev_val_loss = val_loss;
        data_file.clear();
        data_file.seekg(0, std::ios::beg);
    }
    data_file.close();
}

void LogisticRegression::predict() {
    assert(!data_path.empty() && "Predicting requires data path");
    std::string line;
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        std::cout << "(predict) Unable to open file" << std::endl;
        throw 3;
    }
    getline(data_file, line); // Skip header line
    while (getline(data_file, line)) {
        std::vector<std::string> parsed_csv_line = parse_csv_line(&line);
        if (contains_label) {
            parsed_csv_line.pop_back();
        }
        std::vector<float> feature_values = normalize_data(process_line(parsed_csv_line), info.min, info.max);
        float prediction = make_prediction(feature_values);
        std::cout << prediction << std::endl;
    }
    data_file.close();
}

void LogisticRegression::evaluate() {
    assert(!data_path.empty() && "Evaluating requires data path");
    assert(contains_label && "Evaluating requires labeled data");
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;
    std::string line;
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        std::cout << "(evaluate) Unable to open file" << std::endl;
        throw 3;
    }
    getline(data_file, line); // Skip header line
    float total_loss = 0.0f;
    int count = 0;
    while (getline(data_file, line)) {
        std::vector<float> processed_line = process_line(parse_csv_line(&line));
        std::vector<float> feature_values = normalize_data(get_feature_values(processed_line), info.min, info.max);
        float label_value = get_label_value(processed_line);
        float prediction = make_prediction(feature_values);
        int classification = classify(prediction);
        // Confusion matrix
        if (classification == 1 && label_value == 1) {
            true_positive++;
        }
        else if (classification == 0 && label_value == 0) {
            true_negative++;
        }
        else if (classification == 1 && label_value == 0) {
            false_positive++;
        }
        else if (classification == 0 && label_value == 1) {
            false_negative++;
        }
        // Calculate loss
        total_loss += loss(prediction, label_value);
        count++;
    }
    data_file.close();
    std::cout << "Average Loss: " << total_loss / count << std::endl;
    std::cout << "True Positive: " << true_positive << std::endl;
    std::cout << "True Negative: " << true_negative << std::endl;
    std::cout << "False Positive: " << false_positive << std::endl;
    std::cout << "False Negative: " << false_negative << std::endl;
    std::cout << "Accuracy: " << (true_positive + true_negative) / static_cast<float>(count) * 100.0f << "%" << std::endl;
    std::cout << "AUC: " << static_cast<float>(true_positive) / (true_positive + false_negative) << std::endl;
}

/**
 * @return false if model info is given, true if only params are given
 */
void LogisticRegression::handle_params(const std::string &params_path) {
    empty_model = true; // Assume model info is not given
    std::vector<std::string> raw_params;
    std::vector<std::string> raw_info;
    std::string line;
    std::ifstream params_file(params_path);
    if (!params_file.is_open()) {
        std::cout << "Unable to open params/model file" << std::endl;
        throw 3;
    }
    // Handle model parameters
    raw_params.reserve(PARAMS_SIZE());
    while (getline(params_file, line)) {
        if (raw_params.size() >= PARAMS_SIZE()) {
            empty_model = false; // Model info is given
            break;
        }
        raw_params.push_back(line);
    }
    if (raw_params.size() != PARAMS_SIZE()) {
        std::cout << "Invalid number of parameters" << std::endl;
        throw 1;
    }
    try {
        params.learning_rate = std::stof(raw_params[0]);
        params.batch_size = std::stoi(raw_params[1]);
        params.num_epochs = std::stoi(raw_params[2]);
        params.threshold = std::stof(raw_params[3]);
        params.outlier_std = std::stof(raw_params[4]);
        params.early_stopping_threshold = std::stof(raw_params[5]);
    }
    catch (const std::invalid_argument &ia) {
        std::cout << "(handle_params parameters) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range &oor) {
        std::cout << "(handle_params parameters) Out of range: " << oor.what() << std::endl;
        throw 2;
    }
    // Handle model data if given
    if (empty_model) {
        params_file.close();
        return;
    }
    raw_info.reserve(INFO_SIZE());
    do {
        raw_info.push_back(line);
    } while (getline(params_file, line) && raw_info.size() < INFO_SIZE());
    if (raw_info.size() != INFO_SIZE()) {
        std::cout << "Invalid model info: " << raw_info.size() << std::endl;
        std::cout << line << std::endl;
        throw 1;
    }
    try {
        info.weights = process_line(parse_csv_line(&raw_info[0]));
        info.bias = std::stof(raw_info[1]);
        info.min = process_line(parse_csv_line(&raw_info[2]));
        info.max = process_line(parse_csv_line(&raw_info[3]));
    }
    catch (const std::invalid_argument &ia) {
        std::cout << "(handle_params model info) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range &oor) {
        std::cout << "(handle_params model info) Out of range: " << oor.what() << std::endl;
        throw 2;
    }
    params_file.close();
}

float LogisticRegression::loss(const float prediction, const float actual) {
    return (-actual * log(prediction) - (1 - actual) * log(1 - prediction));
}

inline float LogisticRegression::loss_gradient(const float prediction, const float actual) {
    return (prediction - actual);
}

inline std::vector<float> LogisticRegression::loss_gradient(const std::vector<float> &predictions, const std::vector<float> &actuals) {
    return subtract(predictions, actuals);
}

float LogisticRegression::make_prediction(const std::vector<float> &feature_values) {
    return sigmoid(dot(info.weights, feature_values) + info.bias);
}

inline int LogisticRegression::classify(const float prediction) {
    return (prediction >= params.threshold) ? 1 : 0;
}
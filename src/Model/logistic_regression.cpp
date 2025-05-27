#include "model.h"
#include "../Utils/utils.h"
#include "../Utils/math_utils.h"
#include "../Utils/data_utils.h"
#include "logistic_regression.h"

LogisticRegression::LogisticRegression(const std::string &params_path) {
    handle_params(params_path);
}

void LogisticRegression::load_data(const std::string &data_path, bool contains_label){
    // Prepare validation data for early stopping
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        throw std::runtime_error("(load_data) Unable to open data file");
    }

    std::ofstream tmp_file(tmp_path, std::ios::trunc);
    if (!tmp_file.is_open()) {
        throw std::runtime_error("(load_data) Unable to open tmp file");
    }
    // Read data file and write every 5th line to tmp file
    std::string line;
    getline(data_file, line); // Skip header line
    unsigned int line_num = 0;
    unsigned int i_outlier = 0;
    while (getline(data_file, line)) {
        if (outliers.find(i_outlier) != outliers.end()) {
            i_outlier++;
            // Skip outliers
            continue;
        }
        if (++line_num % 5 == 0) {
            // Print line to "./tmp/tmp_train.txt" for early stopping
            tmp_file << line << std::endl;
        }
    }
    tmp_file.close();
    data_file.close();

    // Initilize model parameters and info
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
}

void LogisticRegression::save_model(const std::string &model_path) const {
    std::ofstream model_file(model_path);
    if (!model_file.is_open()) {
        throw std::runtime_error("(save_model) Unable to open model file");
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

void LogisticRegression::print_model() const {
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
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        throw std::runtime_error("(train) Unable to open data file");
    }
    std::ifstream val_file(tmp_path);
    if (!val_file.is_open()) {
        throw std::runtime_error("(train) Unable to open tmp file for early stopping");
    }
    std::string line;
    float prev_val_loss = MAXFLOAT;
    for (int epoch = 0; epoch < params.num_epochs; ++epoch) {
        // Training
        std::cout<< "Epoch: " << epoch + 1 << "/" << params.num_epochs << std::endl;
        process_epoch(data_file);
        data_file.clear();
        data_file.seekg(0, std::ios::beg);
        if (epoch % 5 != 0) {
            continue;
        }
        // Early stopping (checked every 5 epochs)
        float val_loss = get_val_loss(val_file, prev_val_loss);
        val_file.clear();
        val_file.seekg(0, std::ios::beg);
        if (val_loss > prev_val_loss * (1.0f + params.early_stopping_threshold)) {
            std::cout << "Early stopping at epoch " << epoch << " with validation loss: " << val_loss << std::endl;
            break;
        }
        prev_val_loss = val_loss;
    }
    val_file.close();
    data_file.close();
}

void LogisticRegression::predict() {
    std::string line;
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        throw std::runtime_error("(predict) Unable to open data file");
    }
    getline(data_file, line); // Skip header line
    while (getline(data_file, line)) {
        std::vector<std::string> parsed_csv_line = parse_csv_line(line);
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
    std::string line;
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        throw std::runtime_error("(evaluate) Unable to open data file");
    }
    getline(data_file, line); // Skip header line
    float total_loss = 0.0f;
    unsigned int count = 0;
    std::vector<int> label_values;
    std::vector<float> predictions;
    std::vector<int> classifications;
    while (getline(data_file, line)) {
        std::vector<float> feature_values = normalize_data(get_feature_values(line), info.min, info.max);
        label_values.push_back(get_label_value(line));
        predictions.push_back(make_prediction(feature_values));
        classifications.push_back(classify(predictions.back()));
        // Calculate loss
        total_loss += loss(predictions.back(), label_values.back());
        count++;
    }
    data_file.close();
    unsigned int* conf_matrix = confusion_matrix(classifications, label_values);
    std::cout << "Average Loss: " << total_loss / count << std::endl;
    std::cout << "True Positive: " << conf_matrix[0] << std::endl;
    std::cout << "True Negative: " << conf_matrix[1] << std::endl;
    std::cout << "False Positive: " << conf_matrix[2] << std::endl;
    std::cout << "False Negative: " << conf_matrix[3] << std::endl;
    std::cout << "Accuracy: " << accuracy(classifications, label_values) << std::endl;
    std::cout << "Precision: " << precision(classifications, label_values) << std::endl;
    std::cout << "Recall: " << recall(classifications, label_values) << std::endl;
}

/**
 * @return false if model info is given, true if only params are given
 */
void LogisticRegression::handle_params(const std::string &params_path) {
    this->empty_model = true; // Assume model info is not given
    std::vector<std::string> raw_params;
    std::vector<std::string> raw_info;
    std::string line;
    std::ifstream params_file(params_path);
    if (!params_file.is_open()) {
        throw std::runtime_error("(handle_params) Unable to open params/model file");
    }
    // Handle model parameters
    raw_params.reserve(PARAMS_SIZE());
    while (getline(params_file, line)) {
        if (raw_params.size() >= PARAMS_SIZE()) {
            this->empty_model = false; // Model info is given
            break;
        }
        raw_params.push_back(line);
    }
    if (raw_params.size() != PARAMS_SIZE()) {
        throw std::runtime_error("(handle_params) Invalid number of parameters");
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
        throw std::runtime_error("(handle_params parameters) Invalid argument");
    }
    catch (const std::out_of_range &oor) {
        throw std::runtime_error("(handle_params parameters) Out of range");
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
        throw std::runtime_error("(handle_params model info) Invalid number of model info lines");
    }
    try {
        info.weights = process_line(parse_csv_line(raw_info[0]));
        info.bias = std::stof(raw_info[1]);
        info.min = process_line(parse_csv_line(raw_info[2]));
        info.max = process_line(parse_csv_line(raw_info[3]));
    }
    catch (const std::invalid_argument &ia) {
        throw std::runtime_error("(handle_params model info) Invalid argument");
    }
    catch (const std::out_of_range &oor) {
        throw std::runtime_error("(handle_params model info) Out of range");
    }
    params_file.close();
}

void LogisticRegression::process_epoch(std::ifstream &data_file) {
    std::string line;
    std::vector<float> label_value_batch(params.batch_size);
    std::vector<float> prediction_batch(params.batch_size);
    std::vector<std::vector<float>> feature_values_batch(params.batch_size);
    getline(data_file, line); // Skip header line
    unsigned int i_batch = 0;
    unsigned int i_outlier = 0;
    while (getline(data_file, line)) {
        if (outliers.find(i_outlier) != outliers.end()) {
            i_outlier++;
            continue;
        }
        const std::vector<float> feature_values = normalize_data(get_feature_values(line), info.min, info.max);
        label_value_batch[i_batch] = get_label_value(line);
        prediction_batch[i_batch] = make_prediction(feature_values);
        feature_values_batch[i_batch] = feature_values;
        i_batch++;
        if (i_batch == params.batch_size) {
            // Adjust weights and bias for the batch
            update_model(prediction_batch, label_value_batch, feature_values_batch);
            i_batch = 0; // Reset batch index
        }
    }
    // Handle the last batch
    if (i_batch > 0) {
        update_model(prediction_batch, label_value_batch, feature_values_batch);
    }
}

void LogisticRegression::update_model(const std::vector<float> &prediction_batch, const std::vector<float> &label_value_batch, const std::vector<std::vector<float>> &feature_values_batch) {
    const std::vector<float> errors = loss_gradient(prediction_batch, label_value_batch);
    for (size_t i = 0; i < prediction_batch.size(); ++i) {
        for (size_t j = 0; j < info.weights.size(); ++j) {
            this->info.weights[j] -= this->params.learning_rate * errors[i] * feature_values_batch[i][j];
        }
        this->info.bias -= this->params.learning_rate * errors[i];
    }
}

float LogisticRegression::get_val_loss(std::ifstream &val_file, const float prev_val_loss) const {
    std::string line;
    float val_loss = 0.0f;
    unsigned int val_line_num = 0;
    while (getline(val_file, line)) {
        std::vector<float> feature_values = normalize_data(get_feature_values(line), info.min, info.max);
        float label_value = get_label_value(line);
        float prediction = make_prediction(feature_values);
        val_loss += loss(prediction, label_value);
        val_line_num++;
    }
    val_loss /= val_line_num;
    return val_loss;;
}

float LogisticRegression::loss(const float prediction, const float actual) const {
    const float eps = 1e-7f;
    const float p = std::clamp(prediction, eps, 1.0f - eps);
    return (-actual * log(p) - (1 - actual) * log(1 - p));
}

inline float LogisticRegression::loss_gradient(const float prediction, const float actual) const {
    return (prediction - actual);
}

inline std::vector<float> LogisticRegression::loss_gradient(const std::vector<float> &predictions, const std::vector<float> &actuals) const {
    return subtract(predictions, actuals);
}

float LogisticRegression::make_prediction(const std::vector<float> &feature_values) const {
    return sigmoid(dot(this->info.weights, feature_values) + this->info.bias);
}

inline int LogisticRegression::classify(const float prediction) const {
    return (prediction >= this->params.threshold) ? 1 : 0;
}
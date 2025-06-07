#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <cassert>
#include <filesystem>
#include <random>
#include <algorithm>

#include "model.h"
#include "../Utils/utils.h"
#include "../Utils/math_utils.h"
#include "../Utils/data_utils.h"
#include "../Utils/metric_utils.h"

class LogisticRegression : public Model {
public:
    LogisticRegression() = delete; // Prevent default constructor
    LogisticRegression(const std::string &params_path);
    void load_data(const std::string &data_path, bool contains_label) override;
    void save_model(const std::string &model_path) const override;
    void print_model() const override;
    void train() override;
    void predict() const override;
    void evaluate() const override;

protected:
    void handle_params(const std::string &params_path) override;
    float loss(const float predicted, const float actual) const override;
    float loss_gradient(const float predicted, const float actual) const override;
    std::vector<float> loss_gradient(const std::vector<float> &predictions, const std::vector<float> &actuals) const override;
    float make_prediction(const std::vector<float> &feature_values) const;
    int classify(const float prediction) const;
    int classify(const float prediction, const float threshold) const;
    // learning_rate, batch_size, num_epochs, threshold, outlier_std, early_stopping_threshold
    inline int PARAMS_SIZE() const override {return 6;}
    // weights, bias, min, max
    inline int INFO_SIZE() const override {return 4;} 

private:
    struct Params {
        float learning_rate;
        int batch_size;
        int num_epochs;
        float threshold;
        float outlier_std;
        float early_stopping_threshold;
    } params;
    struct Info {
        std::vector<float> weights;
        float bias;
        std::vector<float> min;
        std::vector<float> max;
    } info;

    bool empty_model;
    std::unordered_set<int> outliers;
    const std::filesystem::path tmp_path = "./tmp/tmp_train.txt";
    void process_epoch(std::ifstream &data_file);
    void update_model(const std::vector<float> &prediction_batch, const std::vector<float> &label_value_batch, const std::vector<std::vector<float>> &feature_values_batch);
    float get_val_loss(std::ifstream &val_file, const float prev_val_loss) const;
    float auc(const std::vector<float> &predictions, const std::vector<int> &label_values) const;
};

#endif // LOGISTIC_REGRESSION_H
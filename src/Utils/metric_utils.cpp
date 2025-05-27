#include "metric_utils.h"

std::vector<unsigned int> confusion_matrix(const std::vector<int> &classifications, const std::vector<int> &label_values) {
    if (classifications.size() != label_values.size()) {
        throw std::invalid_argument("Predictions and actuals must have the same size.");
    }
    
    std::vector<unsigned int> matrix(4, 0); // [TP, FP, FN, TN]
    
    for (size_t i = 0; i < classifications.size(); ++i) {
        if (classifications[i] == 1 && label_values[i] == 1) {
            matrix[0]++; // True Positive
        } else if (classifications[i] == 1 && label_values[i] == 0) {
            matrix[1]++; // False Positive
        } else if (classifications[i] == 0 && label_values[i] == 1) {
            matrix[2]++; // False Negative
        } else if (classifications[i] == 0 && label_values[i] == 0) {
            matrix[3]++; // True Negative
        }
    }
    
    return matrix;
}

float accuracy(const std::vector<int> &classifications, const std::vector<int> &label_values) {
    if (classifications.size() != label_values.size()) {
        throw std::invalid_argument("Predictions and actuals must have the same size.");
    }
    
    unsigned int correct = 0;
    for (size_t i = 0; i < classifications.size(); ++i) {
        if (classifications[i] == label_values[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / classifications.size();
}

float precision(const std::vector<int> &classifications, const std::vector<int> &label_values) {
    if (classifications.size() != label_values.size()) {
        throw std::invalid_argument("Predictions and actuals must have the same size.");
    }
    
    unsigned int true_positive = 0;
    unsigned int false_positive = 0;
    
    for (size_t i = 0; i < classifications.size(); ++i) {
        if (classifications[i] == 1.0f && label_values[i] == 1.0f) {
            true_positive++;
        } else if (classifications[i] == 1.0f && label_values[i] == 0.0f) {
            false_positive++;
        }
    }
    
    if (true_positive + false_positive == 0) {
        return 0.0f; // Avoid division by zero
    }
    
    return static_cast<float>(true_positive) / (true_positive + false_positive);
}

float recall(const std::vector<int> &classifications, const std::vector<int> &label_values) {
    if (classifications.size() != label_values.size()){
        throw std::invalid_argument("Predictions and actuals must have the same size.");
    }

    unsigned int true_positive = 0;
    unsigned int false_negative = 0;

    for (size_t i = 0; i < classifications.size(); ++i) {
        if (classifications[i] == 1.0f && label_values[i] == 1.0f) {
            true_positive++;
        }
        else if (classifications[i] == 0.0f && label_values[i] == 1.0f) {
            false_negative++;
        }
    }

    if (true_positive + false_negative == 0) {
        return 0.0f; // Avoid division by zero
    }
    
    return static_cast<float>(true_positive) / (true_positive + false_negative);
}
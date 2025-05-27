#ifndef METRIC_UTILS_H
#define METRIC_UTILS_H

#include <iostream>
#include <string>
#include <vector>

unsigned int* confusion_matrix(const std::vector<int> &classifications, const std::vector<int> &label_values);
float accuracy(const std::vector<int> &classifications, const std::vector<int> &label_values);
float precision(const std::vector<int> &classifications, const std::vector<int> &label_values);
float recall(const std::vector<int> &classifications, const std::vector<int> &label_values);
float auc(const std::vector<float> &predictions, const std::vector<float> &actuals);

#endif // METRIC_UTILS_H
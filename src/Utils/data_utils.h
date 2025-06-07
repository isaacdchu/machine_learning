#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <unordered_set>

#include "utils.h"
#include "math_utils.h"

unsigned int get_num_features(const std::string &data_path, bool contains_label);
std::vector<float> normalize_data(const std::vector<float> &data, const std::vector<float> &min, const std::vector<float> &max);
std::vector<float> get_min(const std::string &data_path, bool contains_label);
std::vector<float> get_max(const std::string &data_path, bool contains_label);
std::vector<float> get_min(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers);
std::vector<float> get_max(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers);
std::unordered_set<int> get_outliers(const std::string &data_path, bool contains_label, float std);

#endif // DATA_UTILS_H
#include <unordered_set>

#include "utils.h"
#include "math_utils.h"
#include "data_utils.h"

int get_num_features(const std::string &data_path, bool contains_label);
std::vector<float> normalize_data(const std::vector<float> &data, const std::vector<float> &min, const std::vector<float> &max);
std::vector<float> get_min(const std::string &data_path, bool contains_label);
std::vector<float> get_max(const std::string &data_path, bool contains_label);
std::vector<float> get_min(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers);
std::vector<float> get_max(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers);
std::unordered_set<int> get_outliers(const std::string &data_path, bool contains_label, float std);

int get_num_features(const std::string &data_path, bool contains_label) {
    std::string line;
    std::ifstream data_file(data_path);
    if (!data_file.is_open()) {
        std::cout << "(get_num_features) Unable to open file" << std::endl;
        throw 3;
    }
    getline(data_file, line);
    std::vector<std::string> parsed_line = parse_csv_line(line);
    data_file.close();
    return parsed_line.size() - contains_label;
}

std::vector<float> normalize_data(const std::vector<float> &data, const std::vector<float> &min, const std::vector<float> &max) {
    return divide(subtract(data, min), subtract(max, min));
}

std::vector<float> get_min(const std::string &data_path, bool contains_label) {
    std::vector<float> min_values;
    std::vector<std::vector<float>> data = get_input(&data_path);
    for (size_t i = 0; i < data[0].size() - contains_label; ++i) {
        float min_value = data[0][i];
        for (const auto &row : data) {
            if (row[i] < min_value) {
                min_value = row[i];
            }
        }
        min_values.push_back(min_value);
    }
    return min_values;
}

std::vector<float> get_max(const std::string &data_path, bool contains_label) {
    std::vector<float> max_values;
    std::vector<std::vector<float>> data = get_input(&data_path);
    for (size_t i = 0; i < data[0].size() - contains_label; ++i) {
        float max_value = data[0][i];
        for (const auto &row : data) {
            if (row[i] > max_value) {
                max_value = row[i];
            }
        }
        max_values.push_back(max_value);
    }
    return max_values;
}

std::vector<float> get_min(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers) {
    std::vector<float> min_values;
    std::vector<std::vector<float>> data = get_input(&data_path);
    for (size_t i = 0; i < data[0].size() - contains_label; ++i) {
        float min_value = std::numeric_limits<float>::max();
        for (size_t j = 0; j < data.size(); ++j) {
            if (outliers.find(j) == outliers.end()) {
                continue;
            }
            if (data[j][i] < min_value){
                min_value = data[j][i];
            }
        }
        min_values.push_back(min_value);
    }
    return min_values;
}

std::vector<float> get_max(const std::string &data_path, bool contains_label, std::unordered_set<int> &outliers) {
    std::vector<float> max_values;
    std::vector<std::vector<float>> data = get_input(&data_path);
    for (size_t i = 0; i < data[0].size() - contains_label; ++i) {
        float max_value = std::numeric_limits<float>::lowest();
        for (size_t j = 0; j < data.size(); ++j) {
            if (outliers.find(j) == outliers.end()) {
                continue;
            }
            if (data[j][i] > max_value){
                max_value = data[j][i];
            }
        }
        max_values.push_back(max_value);
    }
    return max_values;
}

std::unordered_set<int> get_outliers(const std::string &data_path, bool contains_label, float std) {
    std::unordered_set<int> outliers;
    std::vector<std::vector<float>> data = get_input(&data_path);
    std::vector<float> means(data[0].size() - contains_label, 0.0f);
    std::vector<float> std_devs(data[0].size() - contains_label, 0.0f);

    // Calculate means
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size() - contains_label; ++i) {
            means[i] += row[i];
        }
    }
    for (auto &mean : means) {
        mean /= data.size();
    }

    // Calculate standard deviations
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size() - contains_label; ++i) {
            std_devs[i] += (row[i] - means[i]) * (row[i] - means[i]);
        }
    }
    for (auto &std_dev : std_devs) {
        std_dev = sqrt(std_dev / data.size());
    }

    // Identify outliers
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size() - contains_label; ++j) {
            if (fabs(data[i][j] - means[j]) > std * std_devs[j]) {
                outliers.insert(i); 
                break;
            }
        }
    }

    return outliers;
}
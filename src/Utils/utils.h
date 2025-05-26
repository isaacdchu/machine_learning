#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector<float>> get_input(const std::string *data_path);
float get_label_value(const std::vector<float>& processed_line);
std::vector<float> get_feature_values(const std::vector<float>& processed_lin);
std::vector<std::string> parse_csv_line(std::string* line);
std::vector<float> process_line(std::vector<std::string> line);

template <typename T>
void print_vector(std::vector<T> v) {
    std::string str;
    for (const T &item : v)
    {
        str += std::to_string(item) + ",";
    }
    str.pop_back();
    std::cout << str << std::endl;
}

template <typename T>
void print_vector(const std::vector<T> &v, std::ostream &out) {
    if (v.empty())
        return;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i != v.size() - 1)
            out << ",";
    }
    out << std::endl;
}

#endif // UTILS_H
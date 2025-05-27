#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector<float>> get_input(const std::string *data_path, const bool contains_header=true);
float get_label_value(const std::string &line);
std::vector<float> get_feature_values(const std::string &line);
std::vector<std::string> parse_csv_line(const std::string& line);
std::vector<float> process_line(const std::vector<std::string> line);

template <typename T>
void print_vector(const std::vector<T> &v) {
    std::string str;
    for (const T &item : v) {
        str += std::to_string(item) + ",";
    }
    str.pop_back();
    std::cout << str << std::endl;
}

template <typename T>
void print_vector(const std::vector<T> &v, std::ostream &out) {
    if (v.empty())
        return;
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != v.size() - 1)
            out << ",";
    }
    out << std::endl;
}

#endif // UTILS_H
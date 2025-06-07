#include "utils.h"

std::vector<std::vector<float>> get_input(const std::string* data_path, const bool contains_header) {
    std::vector<std::vector<float>> data;
    std::string line;
    std::ifstream myfile(*data_path);
    if (!myfile.is_open()) {
        std::cout << "(get_input) Unable to open file" << std::endl;
        throw 3;
    }
    if (contains_header) {
        getline(myfile, line); // Skip header line
    }
    while (getline(myfile, line)) {
        std::vector<float> parsed_line;
        parsed_line = process_line(parse_csv_line(line));
        data.push_back(parsed_line);
    }
    return data;
}

float get_label_value(const std::string& line) {
    // find the value in the last column of the line
    size_t pos = line.find_last_of(',', line.length() - 2);
    try {
        return std::stof(line.substr(pos + 1));
    }
    catch (const std::invalid_argument& ia) {
        std::cout << "(get_label_value) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range& oor) {
        std::cout << "(get_label_value) Out of range: " << oor.what() << std::endl;
        throw 2;
    }
}

std::vector<float> get_feature_values(const std::string& line) {
    // returns the feature values (all values except the last one)
    size_t pos = line.find_last_of(',', line.length() - 2);
    std::string processed_line_str = line.substr(0, pos);
    std::vector<float> feature_values;
    size_t start = 0;
    size_t end;
    try {
        while ((end = line.find(',', start)) != std::string::npos) {
            feature_values.push_back(stof(line.substr(start, end - start)));
            start = end + 1;
        }
        feature_values.push_back(stof(line.substr(start)));
        return feature_values;
    }
    catch (const std::invalid_argument& ia) {
        std::cout << "(get_feature_values) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range& oor) {
        std::cout << "(get_feature_values) Out of range: " << oor.what() << std::endl;
        throw 2;
    }
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> parsed_line;
    size_t start = 0;
    size_t end;

    while ((end = line.find(',', start)) != std::string::npos) {
        parsed_line.push_back(line.substr(start, end - start));
        start = end + 1;
    }
    parsed_line.push_back(line.substr(start));

    return parsed_line;
}

std::vector<float> process_line(const std::vector<std::string> line) {
    // converts data into floats
    std::vector<float> processed_line;
    try {
        for (const std::string& item : line) {
            processed_line.push_back(std::stof(item));
        }
        return processed_line;
    }
    catch (const std::invalid_argument& ia) {
        std::cout << "(process_line) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range& oor) {
        std::cout << "Out of range: " << oor.what() << std::endl;
        throw 2;
    }
}
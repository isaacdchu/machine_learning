#include "utils.h"

std::vector<std::vector<float>> get_input(const std::string* data_path) {
    std::vector<std::vector<float>> data = {};
    std::string line;
    std::ifstream myfile(*data_path);
    if (!myfile.is_open()) {
        std::cout << "(get_input) Unable to open file" << std::endl;
        throw 3;
    }
    while (getline(myfile, line)) {
        std::vector<float> parsed_line;
        try {
            parsed_line = process_line(parse_csv_line(&line));
        }
        catch (...) {
            continue;
        }
        data.push_back(parsed_line);
    }
    myfile.close();
    return data;
}

float get_label_value(const std::vector<float>& processed_line) {
    return processed_line.back();
}

std::vector<float> get_feature_values(const std::vector<float>& processed_line) {
    // returns the feature values (all values except the last one)
    std::vector<float> feature_values;
    for (size_t i = 0; i < processed_line.size() - 1; ++i) {
        feature_values.push_back(processed_line[i]);
    }
    return feature_values;
}

std::vector<std::string> parse_csv_line(std::string* line) {
    std::vector<std::string> parsed_line;
    size_t start = 0;
    size_t end;

    while ((end = line->find(',', start)) != std::string::npos) {
        parsed_line.push_back(line->substr(start, end - start));
        start = end + 1;
    }
    parsed_line.push_back(line->substr(start));

    return parsed_line;
}

std::vector<float> process_line(std::vector<std::string> line) {
    // converts data into floats
    std::vector<float> processed_line;
    try {
        for (const std::string& item : line) {
            processed_line.push_back(std::stof(item));
        }
        return processed_line;
    }
    catch (const std::invalid_argument& ia) {
        // std::cout << "(process_line) Invalid argument: " << ia.what() << std::endl;
        throw 1;
    }
    catch (const std::out_of_range& oor) {
        std::cout << "Out of range: " << oor.what() << std::endl;
        throw 2;
    }
}
#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Model {
public:
    Model() {};
    virtual void load_data(const std::string &data_path, bool contains_label) = 0;
    virtual void save_model(const std::string &model_path) = 0;
    virtual void print_model() = 0;
    virtual void train() = 0;
    virtual void predict() = 0;
    virtual void evaluate() = 0;

protected:
    virtual void handle_params(const std::string &params_path) = 0;
    virtual float loss(const float prediction, const float actual) = 0;
    virtual float loss_gradient(const float prediction, const float actual) = 0;
    std::string data_path;
    bool contains_label;
    virtual int PARAMS_SIZE() const = 0;
    virtual int INFO_SIZE() const = 0;
};

#endif // MODEL_H

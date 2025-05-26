#include "math_utils.h"

float dot(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> add(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> subtract(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> multiply(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> divide(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> sigmoid(const std::vector<float>& x);
float sigmoid(float x);

// dot product of two vectors, assumes both vectors are of the same size
float dot(const std::vector<float>& x, const std::vector<float>& y) {
    float result = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        result += (x[i] * y[i]);
    }
    return result;
}

// element-wise addition of two vectors, assumes both vectors are of the same size
std::vector<float> add(const std::vector<float> &x, const std::vector<float> &y) {
    std::vector<float> result;
    for (size_t i = 0; i < x.size(); ++i)
    {
        result.push_back(x[i] + y[i]);
    }
    return result;
}

// element-wise subtraction of two vectors, assumes both vectors are of the same size
std::vector<float> subtract(const std::vector<float> &x, const std::vector<float> &y) {
    std::vector<float> result;
    for (size_t i = 0; i < x.size(); ++i) {
        result.push_back(x[i] - y[i]);
    }
    return result;
}

// element-wise multiplication of two vectors, assumes both vectors are of the same size
std::vector<float> multiply(const std::vector<float> &x, const std::vector<float> &y) {
    std::vector<float> result;
    for (size_t i = 0; i < x.size(); ++i)
    {
        result.push_back(x[i] * y[i]);
    }
    return result;
}

// element-wise division of two vectors, assumes both vectors are of the same size
std::vector<float> divide(const std::vector<float> &x, const std::vector<float> &y) {
    std::vector<float> result;
    for (size_t i = 0; i < x.size(); ++i) {
        if (y[i] == 0) {
            if (x[i] == 0) {
                result.push_back(0); // handle 0/0 case
            }
            else if (x.at(i) > 0) {
                result.push_back(MAXFLOAT); // handle division by zero
            }
            else {
                result.push_back(-MAXFLOAT); // handle division by zero
            }
        }
        else {
            result.push_back(x[i] / y[i]);
        }
    }
    return result;
}

// sigmoid function of single value
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// element-wise sigmoid function of a vector
std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> result;
    for (const auto& val : x) {
        result.push_back(sigmoid(val));
    }
    return result;
}
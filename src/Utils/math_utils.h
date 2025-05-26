#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>
#include <cmath>

float dot(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> add(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> subtract(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> multiply(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> divide(const std::vector<float>& x, const std::vector<float>& y);
std::vector<float> sigmoid(const std::vector<float>& x);
float sigmoid(float x);

#endif // MATH_UTILS_H
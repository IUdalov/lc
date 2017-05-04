#pragma once

#include <string>
#include <functional>
#include <map>
#include <cmath>

namespace lc {

class LossFunction {
public:
    LossFunction(const std::string &name, const std::function<double(double)> &function,
                 const std::function<double(double)>& diff) :
            name_(name),
            function_(function),
            diff_(diff) {}
    LossFunction() = default;

    std::string name() const { return name_; }

    double operator()(double v) const { return function_(v); }

    double diff(double v) const { return diff_(v); }

private:
    std::string name_;
    std::function<double(double)> function_;
    std::function<double(double)> diff_;
};

namespace loss_functions {

const LossFunction X1_2(
        "(1 - x)^1/2",
        [](double x) { return x < 1 ? sqrt(1 - x) : 0.0; },
        [](double x) { return x < 1 ? (-1) / (2 * sqrt(1 - x)) : 0.0; });

const LossFunction X(
        "1 - x",
        [](double x) { return x < 1 ? 1 - x : 0.0; },
        [](double x) { return x < 1 ? -1.0 : 0.0; });

const LossFunction X3_2(
        "(1 - x)^3/2",
        [](double x) { return x < 1 ? pow(1 - x, 3/2) : 0.0; },
        [](double x) { return x < 1 ? (-3) / (2 * sqrt(1 - x)) : 0.0; });

const LossFunction X2(
        "(1 - x)^2",
        [](double x) { return x < 1 ? pow(1 - x, 2) : 0.0; },
        [](double x) { return x < 1 ? (-2) * (1 - x) : 0.0; });

const LossFunction X3(
        "(1 - x)^3",
        [](double x) { return x < 1 ? pow(1 - x, 3) : 0.0; },
        [](double x) { return x < 1 ? (-3) * pow(1 - x, 2) : 0.0; });

const LossFunction X4(
        "(1 - x)^4",
        [](double x) { return x < 1 ? pow(1 - x, 4) : 0.0; },
        [](double x) { return x < 1 ? (-4) * pow(1 - x, 3) : 0.0; });

const LossFunction S(
            "2 * (1  + e^x)^-1",
        [](double x) { return (2) / (1 + exp(x)); },
        [](double x) { return (-2) * exp(x) / pow(exp(x) + 1, 2); });

const LossFunction L(
        "log2(1 + e^-x)",
        [](double x) { return log2(1 + exp(-x)); },
        [](double x) { return (-1) / (log(2) * (1 + exp(x))); });

const LossFunction E(
        "exp(-x)",
        [](double x) { return exp(-x); },
        [](double x) { return (-1) * exp(-x); });

inline LossFunction fromName(const std::string& name) {
    static std::map<std::string, LossFunction> data = {
            {"X1_5", X1_2},
            {"V", X},
            {"X", X},
            {"X3_2", X3_2},
            {"Q", X2},
            {"X2", X2},
            {"X3", X3},
            {"X4", X4},
            {"S", S},
            {"L", L},
            {"E", E}
    };

    return data[name];
}

} } // namespace lc::loss_functions
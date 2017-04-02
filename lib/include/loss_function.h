#pragma once

#include <string>
#include <functional>
#include <cmath>

namespace lc {

class LossFunction {
public:
    LossFunction(const std::string &name, const std::function<double(double)> &function,
                 std::function<double(double)> diff) :
            name_(name),
            function_(function),
            diff_(diff) {}

    LossFunction(const LossFunction &) = default;

    ~LossFunction() = default;

    std::string name() const { return name_; }

    double operator()(double v) const { return function_(v); }

    double diff(double v) const { return diff_(v); }

private:
    std::string name_;
    std::function<double(double)> function_;
    std::function<double(double)> diff_;

private:
    LossFunction() = delete;
};

namespace loss_functions {

const LossFunction V(
        "V",
        [](double m) { return m < 1 ? 1 - m : 0.0; },
        [](double m) { return m < 1 ? -1.0 : 0.0; });

const LossFunction Q(
        "Q",
        [](double m) { return m < 1 ? pow(1 - m, 2) : 0.0; },
        [](double m) { return m < 1 ? (-2) * (1 - m) : 0.0; });

const LossFunction Q3(
        "Q3",
        [](double m) { return m < 1 ? pow(1 - m, 3) : 0.0; },
        [](double m) { return m < 1 ? (-3) * pow(1 - m, 2) : 0.0; });

const LossFunction Q4(
        "Q4",
        [](double m) { return m < 1 ? pow(1 - m, 4) : 0.0; },
        [](double m) { return m < 1 ? (-4) * pow(1 - m, 3) : 0.0; });

const LossFunction L(
        "L",
        [](double m) { return log(1 + exp((-1) * m)); },
        [](double m) { return (-1) / (1 + exp((-1) * m)); });

const LossFunction S(
        "S",
        [](double m) { return 2 / (1 + exp(m)); },
        [](double m) { return (-4) / pow(1 + exp((-1) * m), 2); });

const LossFunction E(
        "E",
        [](double m) { return exp(-m); },
        [](double m) { return -1 * exp(-m); });


}

}
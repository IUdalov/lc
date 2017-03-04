#pragma once

#include <linear.h>

#include <vector>
#include <string>
#include <functional>

namespace lc {

class Kernel {
public:
    Kernel(const std::string& name, std::function<double(const std::vector<double>&, const std::vector<double>&)> kernel):
            name_(name),
            kernel_(kernel){}
    Kernel(const Kernel&) = default;
    ~Kernel() = default;

    double operator()(const std::vector<double>& l, const std::vector<double>& r) const { return kernel_(l, r); }
    std::string name() const { return name_; }

private:
    std::string name_;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> kernel_;

private:
    Kernel() = delete;
};

namespace kernels {

const Kernel Homogenous1(
        "Homogenous1",
        [](const Vector& a, const Vector& b) { return dot(a, b); });

const Kernel Homogenous2(
        "Homogenous2",
        [](const Vector& a, const Vector& b) { return pow(dot(a, b), 2); });

const Kernel Homogenous3(
        "Homogenous3",
        [](const Vector& a, const Vector& b) { return pow(dot(a, b), 3); });

const Kernel Inhomogenius1(
        "Inhomogenius1",
        [](const Vector& a, const Vector& b) { return dot(a, b) + 1; });

const Kernel Inhomogenius2(
        "Inhomogenius2",
        [](const Vector& a, const Vector& b) { return pow(dot(a, b) + 1, 2); });

const Kernel Inhomogenius3(
        "Inhomogenius3",
        [](const Vector& a, const Vector& b) { return pow(dot(a, b) + 1, 3); });

// Not tested
const Kernel Radial(
        "Radial",
        [](const Vector& a, const Vector& b) { return exp((-1) * pow(dot(a, b), 2)); });

// Not tested
const Kernel GaussianRadial(
        "GaussianRadial",
        [](const Vector& a, const Vector& b) { return exp((-0.5) * pow(dot(a, b), 2)); });

// Not tested
const Kernel Hyperbolic(
        "Hyperbolic",
        [](const Vector& a, const Vector& b) { return tanh(1 * dot(a, b) - 1); });


}

}
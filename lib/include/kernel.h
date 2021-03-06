#pragma once

#include <data.h>

#include <vector>
#include <string>
#include <functional>

namespace lc {

typedef std::function<double(const std::vector<double>&, const std::vector<double>&)> KernelFunction;

class Kernel {
public:
    Kernel(const std::string& name, const KernelFunction& kernel):
            name_(name),
            kernel_(kernel) {}
    Kernel() = default;
    Kernel(const Kernel&) = default;

    double operator()(const std::vector<double>& l, const std::vector<double>& r) const { return kernel_(l, r); }
    std::string name() const { return name_; }

private:
    std::string name_;
    KernelFunction kernel_;
};

namespace kernels {

const Kernel Homogenous1(
        "Homogenous1",
        [](const Vector& a, const Vector& b) { return internal::dot(a, b); });

const Kernel Homogenous2(
        "Homogenous2",
        [](const Vector& a, const Vector& b) { return pow(internal::dot(a, b), 2); });

const Kernel Homogenous3(
        "Homogenous3",
        [](const Vector& a, const Vector& b) { return pow(internal::dot(a, b), 3); });

const Kernel Inhomogenius1(
        "Inhomogenius1",
        [](const Vector& a, const Vector& b) { return internal::dot(a, b) + 1; });

const Kernel Inhomogenius2(
        "Inhomogenius2",
        [](const Vector& a, const Vector& b) { return pow(internal::dot(a, b) + 1, 2); });

const Kernel Inhomogenius3(
        "Inhomogenius3",
        [](const Vector& a, const Vector& b) { return pow(internal::dot(a, b) + 1, 3); });

const Kernel Radial(
        "Radial",
        [](const Vector& a, const Vector& b) { return exp((-1) * pow(internal::dot(a, b), 2)); });

const Kernel GaussianRadial(
        "GaussianRadial",
        [](const Vector& a, const Vector& b) { return exp((-0.5) * pow(internal::dot(a, b), 2)); });

const Kernel Hyperbolic(
        "Hyperbolic",
        [](const Vector& a, const Vector& b) { return tanh(1 * internal::dot(a, b) - 1); });

inline Kernel fromName(const std::string& name) {
    static std::map<std::string, Kernel> data = {
            {"H1", Homogenous1},
            {"Homogenous1", Homogenous1},
            {"H2", Homogenous2},
            {"Homogenous2", Homogenous2},
            {"H3", Homogenous3},
            {"Homogenous3", Homogenous3},
            {"I1", Inhomogenius1},
            {"Inhomogenius1", Inhomogenius1},
            {"I2", Inhomogenius2},
            {"Inhomogenius2", Inhomogenius2},
            {"I3", Inhomogenius3},
            {"Inhomogenius3", Inhomogenius3},
            {"Radial", Radial},
            {"RAD", Radial},
            {"GaussianRadial", GaussianRadial},
            {"GRAD", GaussianRadial},
            {"Hyperbolic", Hyperbolic},
            {"HYP", Hyperbolic}
    };

    return data[name];
}

} // namespace kernels
} // namespace lc
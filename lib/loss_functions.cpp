#include <lc.h>

#include <map>
#include <math.h>

inline double V(double m) { return m < 1 ? 1 - m : 0.0; }
inline double diffV(double m) { return m < 1 ? -1.0 : 0.0; }

inline double Q(double m) { return m < 1 ? pow(1 - m, 2) : 0.0; }
inline double diffQ(double m) { return m < 1 ? (-2) * (1 - m)  : 0.0; }

inline double Q3(double m) { return m < 1 ? pow(1 - m, 3) : 0.0; }
inline double diffQ3(double m) { return m < 1 ? (-3) * pow(1 - m, 2) : 0.0; }

inline double Q4(double m) { return m < 1 ? pow(1 - m, 4) : 0.0; }
inline double diffQ4(double m) { return m < 1 ? (-4) * pow(1-m, 3) : 0.0; }

inline double L(double m) { return log(1 + exp((-1)*m)); }
inline double diffL(double m) { return (-1) / (1 + exp((-1)*m)); }

inline double S(double m) { return 2 / (1 + exp(m)); }
inline double diffS(double m) { return (-4) / pow(1 + exp((-1)*m), 2); }

inline double E(double m) { return exp(-m); }
inline double diffE(double m) { return -1 * exp(-m); }

namespace lc {
    const Function& lossFunctionRaw(LossFunction lf) noexcept {
        const static std::map<LossFunction, Function> data = {
                {LossFunction::V,  V},
                {LossFunction::Q,  Q},
                {LossFunction::Q3, Q3},
                {LossFunction::Q4, Q4},
                {LossFunction::S,  S},
                {LossFunction::L,  L},
                {LossFunction::E,  E},
        };
        return data.at(lf);
    }

    const Function& lossFunctionDiff(LossFunction lf) noexcept {
        const static std::map<LossFunction, Function> data = {
                {LossFunction::V,  diffV},
                {LossFunction::Q,  diffQ},
                {LossFunction::Q3, diffQ3},
                {LossFunction::Q4, diffQ4},
                {LossFunction::S,  diffS},
                {LossFunction::L,  diffL},
                {LossFunction::E,  diffE},
        };
        return data.at(lf);
    }

    LossFunction lossFunctionByName(const std::string &name) {
        const static std::map<std::string, LossFunction> data = {
                {"V", LossFunction::V},
                {"Q", LossFunction::Q},
                {"Q2", LossFunction::Q},
                {"Q3", LossFunction::Q3},
                {"Q4", LossFunction::Q4},
                {"S", LossFunction::S},
                {"L", LossFunction::L},
                {"E", LossFunction::E},
        };
        return data.at(name);
    }

    std::string lossFunctionToName(LossFunction lf) noexcept {
        const static std::map<LossFunction, std::string> data = {
                {LossFunction::V, "V"},
                {LossFunction::Q, "Q"},
                {LossFunction::Q, "Q2"},
                {LossFunction::Q3, "Q3"},
                {LossFunction::Q4, "Q4"},
                {LossFunction::S, "S"},
                {LossFunction::L, "L"},
                {LossFunction::E, "E"},
        };
        return  data.at(lf);
    }

    const KernelFunction& kernelRaw(Kernel k) noexcept {
        const static std::map<Kernel, KernelFunction> data {
                {Kernel::Homogenous1, [](const Vector& a, const Vector& b) {
                    return dot(a, b);
                }},
                {Kernel::Homogenous2, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b), 2);
                }},
                {Kernel::Homogenous3, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b), 3);
                }},
                {Kernel::Inhomogenius1, [](const Vector& a, const Vector& b) {
                    return dot(a, b) + 1;
                }},
                {Kernel::Inhomogenius3, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b) + 1, 2);
                }},
                {Kernel::Inhomogenius3, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b) + 1, 3);
                }},
                {Kernel::Radial, [](const Vector& a, const Vector& b) {
                    return exp((-1) * pow(dot(a, b), 2)); // not tested
                }},
                {Kernel::GaussianRadial, [](const Vector& a, const Vector& b) {
                    return exp((-0.5) * pow(dot(a, b), 2)); // not tested
                }},
                {Kernel::Hyperbolic, [](const Vector& a, const Vector& b) {
                    return tanh(1 * dot(a, b) - 1); // not tested
                }}};
        return data.at(k);
    }

    Kernel kernelByName(const std::string& name) {
        const static std::map<std::string, Kernel> data {
                {"Homogenous1",     Kernel::Homogenous1},
                {"Homogenous2",     Kernel::Homogenous2},
                {"Homogenous3",     Kernel::Homogenous3},
                {"Inhomogenius1",   Kernel::Inhomogenius1},
                {"Inhomogenius2",   Kernel::Inhomogenius2},
                {"Inhomogenius3",   Kernel::Inhomogenius3},
                {"Radial",          Kernel::Radial},
                {"GaussianRadial",  Kernel::GaussianRadial},
                {"Hyperbolic",      Kernel::Hyperbolic}
        };
        return data.at(name);
    }

    std::string kernelToName(Kernel k) noexcept {
        const static std::map<Kernel, std::string> data {
                {Kernel::Homogenous1,   "Homogenous1"},
                {Kernel::Homogenous2,   "Homogenous2"},
                {Kernel::Homogenous3,   "Homogenous3"},
                {Kernel::Inhomogenius1, "Inhomogenius1"},
                {Kernel::Inhomogenius2, "Inhomogenius2"},
                {Kernel::Inhomogenius3, "Inhomogenius3"},
                {Kernel::Radial,        "Radial"},
                {Kernel::GaussianRadial,"GaussianRadial"},
                {Kernel::Hyperbolic,    "Hyperbolic"},
        };
        return data.at(k);
    }
}
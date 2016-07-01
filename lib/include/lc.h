#pragma once

#include <string>
#include <vector>
#include <functional>

namespace lc {
    std::string getVersion();

    typedef std::vector<double> Vector;
    typedef std::vector<Vector> Objects;
    typedef std::function<double(double)> Function;
    typedef std::function<double(const Vector&, const Vector&)> KernelFunction;


    struct Info {
        size_t objects;
        size_t features;
        size_t steps;
        double c;
        double precision;
        Vector w;

        Info();
    };

    enum class LossFunction {
        V, Q, Q3, Q4, S, L, E
    };

    enum class Kernel {
        Homogenous1,
        Homogenous2,
        Homogenous3,
        Inhomogenius1,
        Inhomogenius2,
        Inhomogenius3,
        Radial,
        GaussianRadial,
        Hyperbolic
    };

    class Model {
    public:
        Model();
        ~Model();

        Info train(
                const Objects& objects,
                const Vector& classes,
                bool skipBayes = false,
                bool skipScale = false);
        int predict(const Vector&) const;

        void lossFunction(LossFunction) noexcept;
        LossFunction lossFunction() noexcept;

        void kernel(Kernel) noexcept;
        Kernel kernel() noexcept;

        void c(double c) noexcept;
        double c() noexcept;

        void maximumStepsNumber(size_t steps) noexcept;
        size_t maximumStepsNumber() noexcept;

        void precision(double precision) noexcept;
        double precision() noexcept;

        void save(const std::string& path);
        void load(const std::string& path);

        void classifier(const Vector &);
        const Vector& classifier() const;

        void margins(const Vector &m);
        const Vector& margins() const;

        const Info& i();

    public:
        void bayes(const Objects& objects, const Vector& classes);
        void toMargins(const Objects& x, const Vector& y);
        void toClassifier(const Objects& x, const Vector& y);

    private:
        Vector w_;
        Vector margins_;

        LossFunction lf_;
        Kernel k_;

        double c_;
        size_t maximumSteps_;
        double precision_;

        Info i_;
    };

    void scaleData(Objects& x, double scaleValue, Vector& factor, Vector& offset);
    void unscaleVector(Vector& v, const Vector& factor, const Vector& offset);
    void createCache(const Objects& x, const Vector& y, KernelFunction f, std::vector<std::vector<double>>& cache);

    const Function& lossFunctionRaw(LossFunction lf) noexcept;
    const Function& lossFunctionDiff(LossFunction lf) noexcept;
    LossFunction lossFunctionByName(const std::string &name);
    std::string lossFunctionToName(LossFunction lf) noexcept;

    const KernelFunction& kernelRaw(Kernel k) noexcept;
    Kernel kernelByName(const std::string& name);
    std::string kernelToName(Kernel k) noexcept;

    // Malicious --------------------------------------------------------------
    double dot(const Vector& lf, const Vector& rf);
    double length(const Vector& data);
    double distance(const Vector& v1, const Vector& v2);
    bool compare(double a, double b);
}

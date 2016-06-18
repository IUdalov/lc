#pragma once

#include <string>
#include <vector>
#include <functional>
#include <math.h>

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
        //Vector oldW;
        //bool wasDifused;
        //size_t defused;

        Info();
    };

    enum class LossFunction {
        V, Q, Q3, Q4, S, L, E
    };

    enum class Kernel {
        Homogenous1,
        //Homogenous2,
        Homogenous3,
        Inhomogenius1,
        //Inhomogenius2,
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
                bool skipScale = false,
                bool skipDefuse = true);
        int predict(const Vector&) const;

        void setLossFunction(const Function& _lf, const Function& _diff) noexcept;
        void lossFunction(LossFunction) noexcept;
        LossFunction lossFunction() noexcept;

        void kernel(Kernel) noexcept;
        Kernel kernel() noexcept;

        void setC(double c) noexcept;
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

        void magrins(const Vector &m);
        const Vector& margins() const;

        const Info& i();

    public:
        void bayes(const Objects& objects, const Vector& classes);
        void toMargins(const Objects& x, const Vector& y);
        void toClassifier(const Objects& x, const Vector& y);
        void defuse(Objects& x, Vector& y);

    private:
        Vector w_;
        Vector margins_;

        LossFunction lf_;
        Function lfRaw_;
        Function diffRaw_;

        Kernel k_;

        double c_;
        size_t maximumSteps_;
        double precision_;

        Info i_;
    };

    void scaleData(Objects& x, double scaleValue, Vector& factor, Vector& offset);
    void unscaleVector(Vector& v, const Vector& factor, const Vector& offset);
    //void checkData(const Objects& o, const Vector& c);
    void createCache(const Objects& x, const Vector& y, KernelFunction f, std::vector<std::vector<double>>& cache);

    // Loss functions ---------------------------------------------------------
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

    const Function& lossFuncionRaw(LossFunction lf) noexcept;
    const Function& lossFunctionDiff(LossFunction lf) noexcept;
    LossFunction lossFuncionByName(const std::string& name);
    std::string lossFunctionToName(LossFunction lf) noexcept;

    const KernelFunction& kernelRaw(Kernel k) noexcept;
    Kernel kernelByName(const std::string& name) noexcept;
    std::string kernelToName(Kernel k) noexcept;

    // Malicious --------------------------------------------------------------
    double dot(const Vector& lf, const Vector& rf);
    double length(const Vector& data);
    double distance(const Vector& v1, const Vector& v2);
    bool isSame(double a, double b);
}

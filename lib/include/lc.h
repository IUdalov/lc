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


    struct Info {
        std::string descr;
        size_t objects;
        size_t features;
        size_t steps;
        double c;
        double precision;
        double errorsBefore;
        double errorsAfter;
        Vector w;
        Vector oldW;

        Info();
    };

    class Model {
    public:
        Model();
        ~Model();

        void setData(const Objects& objects_, const Vector& classes_);

        void setLossFunction(const Function& _lf, const Function& _diff) noexcept;
        void setC(double c) noexcept;
        void setMaximumStepsNumber(size_t steps) noexcept;
        void setPrecision(double precision) noexcept;

        Info train(bool skipBayes = false);
        int predict(const Vector&) const;

        void save(const std::string& path);
        void load(const std::string& path);

        void bayes();
        void toMargins();
        void toClassifier();

    public:
        void setClassifier(const Vector&);
        const Vector& getClassifier() const;

        void setMargins(const Vector&);
        const Vector& getMargins() const;

    private:
        Vector w;       // classifier
        Vector margins;

        Objects x;
        Vector y;       // classes

        Function lf;
        Function diff;
        double c;
        size_t maximumSteps;
        double precision;
    };

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

    // Malicious --------------------------------------------------------------
    double dot(const Vector& lf, const Vector& rf);
    double length(const Vector& data);
    double distance(const Vector& v1, const Vector& v2);
}

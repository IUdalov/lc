#pragma once

#include <loss_function.h>
#include <kernel.h>
#include <scaler.h>
#include <bayes.h>

#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <functional>

namespace lc {

inline std::string getVersion() { return "0.1.0"; };

class Model {
public:
    Model();

    void train(const Problem&);
    double predict(const Vector&) const;

    void approximation(Distribution d) { approximation_ = d; }
    Distribution approximation() const { return approximation_; }

    void lossFunction(const LossFunction& lf) { lf_ = lf; };
    const LossFunction& lossFunction() const { return lf_; };

    void kernel(const Kernel& k) { k_ = k; };
    const Kernel& kernel() const { return k_; };

    void c(double c) { c_ = c; };
    double c() const { return c_; };

    void maximumStepsNumber(size_t steps) { maximumSteps_ = steps; };
    size_t maximumStepsNumber() const { return maximumSteps_; };

    void precision(double precision) { precision_ = precision; }
    double precision() const { return precision_; };

    void classifier(const Vector& w) { w_ = w; };
    const Vector& classifier() const { return w_; };

    void margins(const Vector& m) { margins_ = m; };
    const Vector& margins() const { return margins_; };

    void log(std::ostream& out) const;
public:
    void toMargins(const Problem& p);
    void toClassifier(const Problem& p);

private:
    Distribution approximation_;

    Vector w_;
    Vector margins_;
    std::unique_ptr<Scaler> scaler_;

    LossFunction lf_;
    Kernel k_;

    double c_;
    size_t maximumSteps_;
    double precision_;

    // For statistic only
    size_t nobjects_;
    size_t nfeatures_;
    double rprecision_;
    size_t step_;

    friend std::ostream& operator<<(std::ostream&, const Model&);
    friend std::istream& operator>>(std::istream&, Model&);
};

std::istream& operator>>(std::istream&, Model&);
std::ostream& operator<<(std::ostream&, const Model&);

} // namespace lc
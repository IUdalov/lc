#pragma once

#include <loss_function.h>
#include <kernel.h>
#include <scaler.h>

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

    operator bool() const { return isGood_; };

    // Setters/getters
    void lossFunction(const LossFunction& lf) { lf_ = lf; };

    const LossFunction& lossFunction() { return lf_; };

    void kernel(const Kernel& k) { k_ = k; };

    const Kernel& kernel() { return k_; };

    void c(double c) { c_ = c; };

    double c() { return c_; };

    void maximumStepsNumber(size_t steps) { maximumSteps_ = steps; };

    size_t maximumStepsNumber() { return maximumSteps_; };

    void precision(double precision) { precision_ = precision; }

    double precision() { return precision_; };

    void classifier(const Vector& w) { w_ = w; };

    const Vector& classifier() const { return w_; };

    void margins(const Vector& m) { margins_ = m; };

    const Vector& margins() const { return margins_; };

    void log(std::ostream& out);

public:
    void toMargins(const Problem& p);

    void toClassifier(const Problem& p);

private:
    bool isGood_;
    bool oldBayes_;
    bool invertClassifier_;
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

    friend std::ostream& operator<<(std::ostream& out, const Model&);

    friend std::istream& operator>>(std::istream& in, Model&);
};

std::ostream& operator<<(std::ostream& out, const Model&);

std::istream& operator<<(std::istream& in, Model&);

} // namespace lc
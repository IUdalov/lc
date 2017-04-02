#pragma once

#include <loss_function.h>
#include <kernel.h>

#include <string>
#include <vector>
#include <ostream>
#include <functional>

namespace lc {

std::string getVersion();

class Entry {
public:
    Entry(int y, Vector x) : y_(y), x_(x) {}
    Entry(Entry&&) = default;

    int y() const { return y_; }

    Vector& x() { return x_; }
    const Vector& x() const { return x_; }

    size_t size() const { return x_.size(); }

    double& operator[](size_t ind) { return x_[ind]; }
    const double& operator[](size_t ind) const { return x_[ind]; }

    operator bool() const { return y_ > 0; }

    bool operator==(const Entry& other) const {
        return y_ == other.y_ && x_ == other.x_;
    }

    bool operator!=(const Entry& other) const {
        return !operator==(other);
    }

private:
    int y_;
    Vector x_;

private:
    Entry(const Entry&) = delete;
};

inline std::ostream& operator<<(std::ostream& out, const Entry& e) {
    out << (e.y() == 1 ? "+1" : "-1");
    for (size_t i = 0; i < e.x().size(); i++)
        if (e[i] != 0)
            out << " " << (i + 1) << ":" << e[i];

    out << std::endl;
    return out;
}

class Problem {
public:
    Problem() = default;
    Problem(Problem&&) = default;

    Entry& operator[](size_t ind) { return entries_[ind]; }
    const Entry& operator[](size_t ind) const { return entries_[ind]; }

    std::vector<Entry>& entries() { return entries_; };
    const std::vector<Entry>& entries() const { return entries_; };

    void add(Entry e) { entries_.emplace_back(std::move(e)); }
    Problem dup() const {
        Problem dup;
        for(const auto& e : entries_)
            dup.add(Entry(e.y(), e.x()));
        return dup;
    }
private:
    std::vector<Entry> entries_;

private:
    Problem(const Problem&) = delete;
};

inline std::ostream& operator<<(std::ostream& out, const Problem& p) {
    out << "Problem {" << std::endl;
    out << "\tobjects = " << p.entries().size() << std::endl;
    out << "\tfeatures = " << p[0].x().size() << std::endl;
    out << "}" << std::endl;
    return out;
}

struct Info {
    size_t objects;
    size_t features;
    size_t steps;
    double c;
    double precision;
    Vector w;
};

inline std::ostream& operator<<(std::ostream& out, const Info& i) {
    out << "Info {" << std::endl;
    out << "\tobjects = " << i.objects << std::endl;
    out << "\tfeatures = " << i.features << std::endl;
    out << "\tsteps = " << i.steps << std::endl;
    out << "\tc = " << i.c << std::endl;
    out << "\tprecision = " << i.precision << std::endl;
    out << "\tw =";
    for(const auto& o : i.w) out << " " << o;
    out << std::endl << "}" << std::endl;
    return out;
}


class Model {
public:
    Model();
    Model(int argc, char* argv[]);

    Info train(
            const Problem& P,
            bool skipBayes = false,
            bool skipScale = false);
    double predict(const Vector&) const;


    void save(const std::string& path);
    void load(const std::string& path);

    operator bool() const { return isGood_; };

    // Setters/getters
    void lossFunction(const LossFunction& lf) { lf_ = lf; };
    const LossFunction& lossFunction() { return lf_; };

    void kernel(const Kernel& k) { k_ = k; };
    const Kernel& kernel() { return k_; };

    void c(double c) {c_ = c; };
    double c() { return c_; };

    void maximumStepsNumber(size_t steps) {maximumSteps_ = steps; };
    size_t maximumStepsNumber() { return maximumSteps_; };

    void precision(double precision) { precision_ = precision; }
    double precision() { return precision_; };

    void classifier(const Vector& w) { w_ = w; };
    const Vector& classifier() const { return w_; };

    void margins(const Vector& m) { margins_ = m; };
    const Vector& margins() const { return margins_; };

    const Info& i() { return i_; }

public:
    void toMargins(const Problem& p);
    void toClassifier(const Problem& p);

private:
    bool isGood_;
    Vector w_;
    Vector margins_;

    LossFunction lf_;
    Kernel k_;

    double c_;
    size_t maximumSteps_;
    double precision_;

    Info i_;
};

Vector naiveBayes(const Problem& p);
void scaleData(Problem& p, double scaleValue, Vector& factor, Vector& offset);
void unscaleVector(Vector& v, const Vector& factor, const Vector& offset);

}

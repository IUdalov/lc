#include "lc.h"

#include "debug.h"

#include <math.h>
#include <exception>

namespace lc {

namespace {

const size_t DEFAULT_MAXIMUM_STEPS = 100;
const double DEFAULT_PRECISION = 0.00001;

void validate(double v) {
    (void)v;
    //if (isnan(v))
        //throw std::runtime_error("NaN");
}

void validate(const Vector& v, size_t space) {
    if (v.size() != space)
        throw std::runtime_error("Size mismatch");

    for(const auto& e : v)
        validate(e);
}

void validate(const Problem& p) {
    if (p.entries().empty())
        throw std::runtime_error("Problem is empty");

// this checks are quite heavy
#ifndef NDEBUG
    size_t space = p[0].size();
    for(const auto& e : p.entries())
        validate(e.x(), space);

#endif // NDEBUG
}

std::vector<Vector> createCache(const Problem& p, const Kernel &kernel) {
    std::vector<Vector> cache;
    DEBUG << "Caching with kernel " << kernel.name() << std::endl;
    cache.clear();
    size_t l = p.entries().size();
    cache.resize(l, std::vector<double>(l, 0));

    for (auto &c : cache)
        c.shrink_to_fit();

    for (size_t i = 0; i < l; i++)
        for (size_t j = 0; j < l; j++)
            cache[i][j] = p[i].y() * p[j].y() * kernel(p[i].x(), p[j].x());

    DEBUG << "Done" << std::endl;
    return cache;
}

} // namespace

Model::Model()
        : isGood_(true),
          lf_(loss_functions::Q),
          k_(kernels::Homogenous1),
          c_(1),
          maximumSteps_(DEFAULT_MAXIMUM_STEPS),
          precision_(DEFAULT_PRECISION) {
}

Model::Model(int argc, char* argv[])
    : Model() {
    (void)argc;
    (void)argv;
}

Info Model::train(
        const Problem& rawProblem,
        bool skipBayes,
        bool skipScale) {
    validate(rawProblem);
    Problem problem = rawProblem.dup();

    i_.objects = problem.entries().size();
    i_.features = problem[0].x().size();
    i_.c = c_;

    Vector factor;
    Vector offset;

    if (!skipScale)
        scaleData(problem, 1, factor, offset);

    if (!skipBayes)
        w_ = naiveBayes(problem);

    if (!w_.empty())
        toMargins(problem);

    if (margins_.empty())
        margins_.assign(problem.entries().size(), 0.5);

    Vector marginsWas(margins_.size(), 100);
    auto cache = createCache(problem, k_);

    DEBUG << "Train with function " << lf_.name() << " and C " << c_ << std::endl;
    for(i_.steps = 0; i_.steps < maximumSteps_; i_.steps++) {
        i_.precision = distance(margins_, marginsWas);
        validate(i_.precision);
        if (i_.precision < precision_)
            break;
        DEBUG << i_.steps << std::endl;
        marginsWas = margins_;
        for(std::size_t k = 0; k < margins_.size(); k++) {
            double tmp = margins_[k];
            for(std::size_t i = 0; i < margins_.size(); i++) {
                double acc =  (-c_) * lf_.diff(margins_[i]) * cache[i][k];
                tmp += acc;
            }
            margins_[k] = tmp;
        }
    }

    toClassifier(problem);

    if (!skipScale)
        unscaleVector(w_, factor, offset);

    i_.w = w_;
    DEBUG << "Stopped with " << i_;

    return i_;
}

double Model::predict(const Vector& value) const {
    if (w_.empty() || w_.size() != value.size()) {
        throw std::runtime_error("Model was not configured");
    }
    return k_(w_, value);
}

void Model::toMargins(const Problem& p) {
    DEBUG << std::endl;
    validate(p);

    margins_.assign(p.entries().size(), 0);
    for(size_t i = 0; i < p.entries().size(); i++) {
        margins_[i] = k_(w_, p[i].x()) * p[i].y();
    }
}

void Model::toClassifier(const Problem& p) {
    DEBUG << std::endl;
    validate(p);

    size_t n = p[0].x().size();
    w_.assign(n, 0);

    for(size_t i = 0; i < p.entries().size(); i++) {
        for(size_t k = 0; k < n; k++) {
            double diffI = lf_.diff(-margins_[i]);
            w_[k] += (-c_) * diffI * p[i][k] * p[i].y();
        }
    }
}

Vector naiveBayes(const Problem& p) {
    DEBUG << "Build bayes approximation" << std::endl;
    validate(p);

    size_t n = p[0].x().size();

    size_t p1 = 0;
    size_t p2 = 0;
    Vector means1(n, 0.0);
    Vector means2(n, 0.0);
    Vector dis1(n, 0.0);
    Vector dis2(n, 0.0);

    for(size_t j = 0; j < p.entries().size(); j++) {
        for(size_t i = 0; i < n; i++) {
            if (p[j].y() == 1) {
                p1++;
                means1[i] += p[j][i];
                dis1[i] += pow(p[j][i], 2);
            } else {
                p2++;
                means2[i] += p[j][i];
                dis2[i] += pow(p[j][i], 2);
            }
        }
    }

    for(size_t i = n; i < n; i++) {
        means1[i] = means1[i] / static_cast<double>(p1);
        dis1[i] = pow(dis1[i] / static_cast<double>(p1), 2) - pow(means1[i], 2);

        means2[i] = means2[i] / static_cast<double>(p2);
        dis2[i] = pow(dis2[i] / static_cast<double>(p2), 2) - pow(means2[i], 2);
    }

    Vector w(n, 0);
    for(size_t j = 0; j < p.entries().size(); j++) {
        for(size_t i = 0; i < n; i++) {
            double mean = p[j].y() == 1 ? means1[i] : means2[i];
            double dis = p[j].y() == 1 ? dis1[i] : dis2[i];
            w[i] += p[j].y() * (mean / dis);
        }
    }

    return w;
}

void scaleData(Problem& p, double scaleValue, Vector& factor, Vector& offset) {
    DEBUG << "Scaling objects" << std::endl;
    validate(p);

    // let's scale space X -> X'
    // x' = M(x + v)
    // x = (M^-1)x' - v

    size_t l = p.entries().size();
    size_t n = p[0].x().size();

    factor.clear();
    offset.clear();
    factor.resize(n);
    offset.resize(n);

    Vector min(n, std::numeric_limits<double>::max());
    Vector max(n, std::numeric_limits<double>::lowest());

    for(size_t i = 0; i < l; i++) {
        for(size_t j = 0; j < n; j++) {
            if (p[i][j] < min[j]) min[j] = p[i][j];
            if (p[i][j] > max[j]) max[j] = p[i][j];
        }
    }

    for(size_t j = 0; j < n; j++) {
        if (max[j] - min[j] <= std::numeric_limits<double>::epsilon())
            factor[j] = scaleValue;
        else
            factor[j] = scaleValue / (max[j] - min[j]);

        offset[j] = - (max[j] + min[j]) / 2;
    }
    for(size_t i = 0; i < l; i++)
        for(size_t j = 0; j < n; j ++)
            p[i][j] = (p[i][j] + offset[j]) * factor[j];

    validate(p);
}

void unscaleVector(Vector& v, const Vector& factor, const Vector& offset) {
    if (v.empty()) {
        throw std::runtime_error("Empty vector");
    }
    if (v.size() != factor.size() || v.size() != offset.size()) {
        throw std::runtime_error("Vector doesn't fit to scale");
    }
    for(size_t i = 0; i < v.size(); i++) {
        v[i] = v[i] / factor[i] - offset[i];
    }
}

} // namespace lc

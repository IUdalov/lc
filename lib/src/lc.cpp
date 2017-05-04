#include "lc.h"

#include "bayes.h"
#include "debug.h"

#include <cstdlib>
#include <cmath>
#include <exception>

namespace lc {

namespace {

const size_t DEFAULT_MAXIMUM_STEPS = 100;
const double DEFAULT_PRECISION = 0;//std::numeric_limits<double>::epsilon();

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
        : invertClassifier_(std::getenv("LC_INVERT") != nullptr)
        , approximation_(Distribution::Gauss)
        , lf_(loss_functions::X)
        , k_(kernels::Homogenous1)
        , c_(1)
        , maximumSteps_(DEFAULT_MAXIMUM_STEPS)
        , precision_(DEFAULT_PRECISION) {
}

void Model::train(const Problem& rawProblem) {
    validate(rawProblem);
    Problem problem = rawProblem.dup();

    nobjects_ = problem.entries().size();
    nfeatures_ = problem[0].x().size();

    scaler_ = std::make_unique<Scaler>(problem);
    scaler_->apply(problem);

    w_ = naiveBayes(problem, approximation_);
    norm(w_);
    validate(w_, nfeatures_);

    if (maximumSteps_ == 0) {
        DEBUG << "Stopped after naive bayes" << std::endl;
        return;
    }

    toMargins(problem);
    norm(margins_);
    validate(margins_, nobjects_);

    Vector marginsWas(margins_.size(), 100);
    auto cache = createCache(problem, k_);

    DEBUG << "Train with function " << lf_.name() << " and C " << c_ << std::endl;
    for(step_ = 0; step_ < maximumSteps_; step_++) {
        rprecision_ = distance(margins_, marginsWas);
        validate(rprecision_);
        if (rprecision_ < precision_)
            break;

        marginsWas = margins_;
        for(std::size_t k = 0; k < margins_.size(); k++) {
            double tmp = margins_[k];
            for(std::size_t i = 0; i < margins_.size(); i++) {
                double acc =  (-c_) * lf_.diff(margins_[i]) * cache[i][k];
                tmp += acc;
            }
            margins_[k] = tmp;
        }

        // LAST EXPERIMENT WAS WITH COMMENTED NORM
        norm(margins_);
        validate(margins_, nobjects_);
    }

    toClassifier(problem);

    DEBUG << "Stopped with steps: " << step_ << " precision: " << precision_ << std::endl;
}

double Model::predict(const Vector& value) const {
    if (w_.empty() || w_.size() != value.size())
        throw std::runtime_error("Model was not configured");

    Vector copy = value;
    if (scaler_)
        scaler_->apply(copy);
    else
        DEBUG << "Empty scaler!" << std::endl;

    if (invertClassifier_) {
        for(auto& e : copy)
            e = -e;
    }

    return k_(w_, copy);
}

void Model::toMargins(const Problem& p) {
    DEBUG << std::endl;
    validate(p);

    margins_.assign(p.entries().size(), 0);
    printVector("w_ init", w_);
    for(size_t i = 0; i < p.entries().size(); i++) {
        margins_[i] = k_(w_, p[i].x()) * p[i].y();
    }

    printVector("margins_ from w_", margins_);
}

void Model::toClassifier(const Problem& p) {
    DEBUG << std::endl;
    validate(p);

    size_t n = p[0].x().size();
    w_.assign(n, 0);
    printVector("margins_ init", margins_);

    for(size_t i = 0; i < p.entries().size(); i++) {
        for(size_t k = 0; k < n; k++) {
            double diffI = lf_.diff(-margins_[i]);
            w_[k] += (-c_) * diffI * p[i][k] * p[i].y();
        }
    }

    printVector("w_ from margins_", w_);
}

void Model::log(std::ostream& out) {
    out << "Model {" << std::endl;
    out << "\tobjects = " << nobjects_ << std::endl;
    out << "\tfeatures = " << nfeatures_ << std::endl;
    out << "\tsteps = " << step_ << std::endl;
    out << "\tc = " << c_ << std::endl;
    out << "\tprecision = " << rprecision_ << std::endl;
    out << "\tw =";
    for(const auto& o : w_) out << " " << o;
    out << std::endl << "}" << std::endl;
}

} // namespace lc

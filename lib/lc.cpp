#include <lc.h>

#include <math.h>
#include <exception>

const size_t DEFAULT_MAXIMUM_STEPS = 10000;
const double DEFAULT_PRECISION = 0.00001;

namespace lc {
    Info::Info()
        : objects(0),
          features(0),
          steps(0),
          c(-1),
          precision(0),
          w({}) {
    }

    Model::Model()
        : lf_(LossFunction::Q),
          k_(Kernel::Homogenous1),
          c_(1),
          maximumSteps_(DEFAULT_MAXIMUM_STEPS),
          precision_(DEFAULT_PRECISION) {
    }

    Model::~Model() {
    }

    void Model::lossFunction(LossFunction lf) noexcept {
        lf_ = lf;
    }

    LossFunction Model::lossFunction() noexcept {
        return lf_;
    }

    void Model::kernel(Kernel k) noexcept {
        k_ = k;
    }

    Kernel Model::kernel() noexcept {
        return k_;
    }

    void Model::c(double c) noexcept {
        c_ = c;
    }

    double Model::c() noexcept {
        return c_;
    }

    void Model::maximumStepsNumber(size_t n) noexcept {
        maximumSteps_ = n;
    }

    size_t Model::maximumStepsNumber() noexcept {
        return maximumSteps_;
    }

    void Model::precision(double p) noexcept {
        precision_ = p;
    }

    double Model::precision() noexcept {
        return precision_;
    }

    void Model::classifier(const Vector &w) {
        w_ = w;
    }

    const Vector& Model::classifier() const {
        return w_;
    }

    void Model::margins(const Vector &m_) {
        margins_ = m_;
    }

    const Vector& Model::margins() const {
        return margins_;
    }

    const Info& Model::i() {
        return i_;
    }

    Info Model::train(
            const Objects& rawX,
            const Vector& rawY,
            bool skipBayes,
            bool skipScale) {
        if (rawX.empty()) {
            throw std::runtime_error("Problem doesn't have entries");
        }

        if (rawX[0].empty()) {
            throw std::runtime_error("Error obj");
        }

        if (rawX.size() != rawY.size()) {
            throw std::runtime_error("Error size");
        }

        i_.objects = rawX.size();
        i_.features = rawX[0].size();
        i_.c = c_;

        Objects x = rawX;
        Vector y = rawY;

        x.shrink_to_fit();
        y.shrink_to_fit();
        for(auto& it : x) {
            it.shrink_to_fit();
        }

        Vector factor;
        Vector offset;
        if (!skipScale) {
            scaleData(x, 1, factor, offset);
        }

        if (!skipBayes) {
            bayes(x, y);
        }

        if (!w_.empty()) {
            toMargins(x, y);
        }

        if (margins_.size() != x.size()) {
            margins_.assign(x.size(), 0.5);
        }
        Vector marginsWas(margins_.size(), 100);

        margins_.shrink_to_fit();
        marginsWas.shrink_to_fit();

        std::vector<std::vector<double>> cache;
        createCache(x,y, dot,cache);
        for(i_.steps = 0; i_.steps < maximumSteps_; i_.steps++) {
            if (distance(margins_, marginsWas) < precision_) {
                break;
            }

            marginsWas = margins_;
            for(std::size_t k = 0; k < margins_.size(); k++) {
                double tmp = margins_[k];
                Function diffRaw = lossFunctionDiff(lf_);
                for(std::size_t i = 0; i < margins_.size(); i++) {
                    tmp += (-c_) * diffRaw(margins_[i]) * cache[i][k];
                }
                margins_[k] = tmp;
            }
        }

        toClassifier(x, y);

        if (!skipScale) {
            unscaleVector(w_, factor, offset);
        }
        i_.precision = distance(margins_, marginsWas);
        i_.w = w_;

        return i_;
    }

    int Model::predict(const Vector& value) const {
        if (w_.empty() || w_.size() != value.size()) {
            throw std::runtime_error("Model was not configured");
        }
        KernelFunction k(kernelRaw(k_));
        return k(w_, value) >= 0 ? 1 : -1;
    }

    // Probably a lot of errors
    void Model::bayes(const Objects& x, const Vector& y) {
        if (x.empty() || y.empty() || x.size() != y.size()) {
            throw std::runtime_error("Model was not configured");
        }
        size_t n = x[0].size();

        size_t p1 = 0;
        size_t p2 = 0;
        Vector means1(n, 0.0);
        Vector means2(n, 0.0);
        Vector dis1(n, 0.0);
        Vector dis2(n, 0.0);

        for(size_t j = 0; j < x.size(); j++) {
            for(size_t i = 0; i < n; i++) {
                if (y[j] == 1) {
                    p1++;
                    means1[i] += x[j][i];
                    dis1[i] += pow(x[j][i], 2);
                } else {
                    p2++;
                    means2[i] += x[j][i];
                    dis2[i] += pow(x[j][i], 2);
                }
            }
        }

        for(size_t i = n; i < n; i++) {
            means1[i] = means1[i] / static_cast<double>(p1);
            dis1[i] = pow(dis1[i] / static_cast<double>(p1), 2) - pow(means1[i], 2);

            means2[i] = means2[i] / static_cast<double>(p2);
            dis2[i] = pow(dis2[i] / static_cast<double>(p2), 2) - pow(means2[i], 2);
        }

        w_.assign(n, 0);
        for(size_t j = 0; j < x.size(); j++) {
            for(size_t i = 0; i < n; i++) {
                double mean = y[j] == 1 ? means1[i] : means2[i];
                double dis = y[j] == 1 ? dis1[i] : dis2[i];
                w_[i] += y[j] * (mean / dis);
            }
        }
    }

    void Model::toMargins(const Objects& x, const Vector& y) {
        if (w_.empty() || x.empty() || y.empty()) {
            throw std::runtime_error("Unable to create Margins.\nModel was not configured");
        }
        KernelFunction kernel(kernelRaw(k_));
        margins_.assign(x.size(), 0);
        for(size_t i = 0; i < x.size(); i++) {
            margins_[i] = kernel(w_, x[i]) * y[i];
        }
    }

    void Model::toClassifier(const Objects& x, const Vector& y) {
        if (margins_.empty() || x.empty() || y.empty()) {
            throw std::runtime_error("Unable to create Classifier.\nModel was not configured");
        }
        size_t n = x[0].size();
        w_.assign(n, 0);

        Function diffLf = lossFunctionDiff(lf_);
        for(size_t i = 0; i < x.size(); i++) {
            for(size_t k = 0; k < n; k++) {
                double diffI = diffLf(-margins_[i]);
                w_[k] += (-c_) * diffI * x[i][k] * y[i];
            }
        }
    }
}

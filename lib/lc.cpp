#include "lc.h"

#include <math.h>

#include <exception>

const size_t DEFAULT_MAXIMUM_STEPS = 10000;
const double DEFAULT_PRECISION = 0.00000001;

namespace lc {
    Info::Info()
        : descr{"Do not available"},
          objects(0),
          features(0),
          steps(0),
          c(-1),
          precision(0),
          name('%'),
          errorsBefore(-1),
          errorsAfter(-1),
          w({}) {
    }

    Model::Model()
        : lf(Q),
          diff(diffQ),
          c(1),
          maximumSteps(DEFAULT_MAXIMUM_STEPS),
          precision(DEFAULT_PRECISION) {
    }

    Model::~Model() {
    }

    void Model::setLossFunction(const Function& _lf, const Function& _diff) {
        lf = _lf;
        diff = _diff;
    }

    void Model::setData(const Objects &objects_, const Vector &classes_) {
        x = objects_;
        y = classes_;
    }

    void Model::setC(double c_) {
        c = c_;
    }

    void Model::setMaximumStepsNumber(size_t n) {
        maximumSteps = n;
    }
    void Model::setPrecision(double p) {
        precision = p;
    }

    void Model::setClassifier(const Vector& w_) {
        w = w_;
    }

    const Vector& Model::getClassifier() const {
        return w;
    }

    void Model::setMargins(const Vector& m_) {
        margins = m_;
    }

    const Vector& Model::getMargins() const {
        return margins;
    }

    Info Model::train(bool skipBayes) {
        if (x.empty()) {
            throw std::runtime_error("Problem doesn't have entries");
        }

        if (x[0].empty()) {
            throw std::runtime_error("Error obj");
        }

        if (x.size() != y.size()) {
            throw std::runtime_error("Error size");
        }

        Info i;
        i.objects = x.size();
        i.features = x[0].size();
        i.c = c;

        if (!skipBayes) {
            bayes();
        }

        if (!w.empty()) {
            toMargins();
        }

        if (margins.size() != x.size()) {
            margins.assign(x.size(), 0.5);
        }
        Vector marginsWas(margins.size(), 100);

        for(i.steps = 0; i.steps < maximumSteps; i.steps++) {
            if (distance(margins, marginsWas) < precision) {
                break;
            }

            marginsWas = margins;
            for(std::size_t k = 0; k < margins.size(); k++) {
                double tmp = margins[k];

                for(std::size_t i = 0; i < margins.size(); i++) {
                    tmp += (-c) * diff(margins[i]) * y[i] * y[k] * dot(x[i], x[k]);
                }
                margins[k] = tmp;
            }
        }

        toClassifier();

        i.precision = distance(margins, marginsWas);
        i.w = w;
        return i;
    }

    int Model::predict(const Vector& value) const {
        if (w.empty() || w.size() != value.size()) {
            throw std::runtime_error("Model was not configured");
        }
        return dot(w, value) >= 0 ? 1 : -1;
    }

    // TODO: check?
    // Probably a lot of errors
    void Model::bayes() {
        if (x.empty() || y.empty()) {
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

        w.assign(n, 0);
        for(size_t j = 0; j < x.size(); j++) {
            for(size_t i = 0; i < n; i++) {
                double mean = y[j] == 1 ? means1[i] : means2[i];
                double dis = y[j] == 1 ? dis1[i] : dis2[i];
                w[i] += y[j] * (mean / dis);
            }
        }
    }

    void Model::toMargins() {
        if (w.empty() || x.empty() || y.empty()) {
            throw std::runtime_error("Unable to create Margins.\nModel was not configured");
        }

        margins.assign(x.size(), 0);
        for(size_t i = 0; i < x.size(); i++) {
            margins[i] = dot(w, x[i]) * y[i];
        }
    }

    void Model::toClassifier() {
        if (margins.empty() || x.empty() || y.empty()) {
            throw std::runtime_error("Unable to create Classifier.\nModel was not configured");
        }
        size_t n = x[0].size();
        w.assign(n, 0);
        for(size_t i = 0; i < x.size(); i++) {
            for(size_t k = 0; k < n; k++) {
                double diffI = diff(-margins[i]);
                w[k] += (-c) * diffI * x[i][k] * y[i];
            }
        }
    }
}

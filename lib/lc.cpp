#include "lc.h"

#include <math.h>
#include <exception>
#include <fstream>
#include <map>

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
          lfRaw_(Q),
          diffRaw_(diffQ),
          k_(Kernel::Homogenous1),
          c_(1),
          maximumSteps_(DEFAULT_MAXIMUM_STEPS),
          precision_(DEFAULT_PRECISION) {
    }

    Model::~Model() {
    }

    void Model::setLossFunction(const Function& _lf, const Function& _diff) noexcept {
        lfRaw_ = _lf;
        diffRaw_ = _diff;
    }

    void Model::lossFunction(LossFunction lf) noexcept {
        lf_ = lf;
        lfRaw_ = lossFuncionRaw(lf);
        diffRaw_ = lossFunctionDiff(lf);
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

    void Model::setC(double c) noexcept {
        c_ = c;
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

    void Model::magrins(const Vector &m_) {
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
            bool skipScale,
            bool skipDefuse) {
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
            bayes(rawX, rawY);
        }

        if (!skipDefuse) {
            defuse(x, y);
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

                for(std::size_t i = 0; i < margins_.size(); i++) {
                    tmp += (-c_) * diffRaw_(margins_[i]) * cache[i][k];
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
        return dot(w_, value) >= 0 ? 1 : -1;
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

        margins_.assign(x.size(), 0);
        for(size_t i = 0; i < x.size(); i++) {
            margins_[i] = dot(w_, x[i]) * y[i];
        }
    }

    void Model::toClassifier(const Objects& x, const Vector& y) {
        if (margins_.empty() || x.empty() || y.empty()) {
            throw std::runtime_error("Unable to create Classifier.\nModel was not configured");
        }
        size_t n = x[0].size();
        w_.assign(n, 0);
        for(size_t i = 0; i < x.size(); i++) {
            for(size_t k = 0; k < n; k++) {
                double diffI = diffRaw_(-margins_[i]);
                w_[k] += (-c_) * diffI * x[i][k] * y[i];
            }
        }
    }

    // TODO: this implementation spoilers initial task
    // Just disabled for now
    void  Model::defuse(Objects& x, Vector& y) {
        return;
        /*
        if (oldX.empty() || oldY.empty() || oldX.size() != oldY.size() || oldX[0].empty()) {
            throw std::runtime_error("Objects do not present for defusing");
        }
        if (w_.empty()) {
            newX = oldX;
            newY = oldY;
            return;
        }

        auto lfDiff = lossFunctionDiff(lf_);
        for(size_t i = 0; i < oldX.size(); i++) {
            if (!isSame(lfDiff(dot(w_, oldX[i]) * oldY[i]), 0)) {
                newX.push_back(oldX[i]);
                newY.push_back(oldY[i]);
            }
        }

        i_.defused = oldX.size() - newX.size();

        if (oldX.size() == newX.size()) {
            return;
        }

        if (oldX.size() / 2 < (newX.size() + 1)) {
            newX = oldX;
            newY = oldY;
            return;
        }
        i_.wasDifused = true;*/
    }

    void Model::save(const std::string& path) {
        if (w_.empty()) {
            throw std::runtime_error("Model was not configured");
        }
        std::ofstream out;
        out.open(path);

        for(auto it : w_) {
            out << it << std::endl;
        }

        out.close();
    }

    void Model::load(const std::string& path) {
        w_.clear();
        std::ifstream modelFile(path);
        if (!modelFile) {
            throw std::runtime_error("File " + path + " not found!");
        }
        std::string line;
        while(std::getline(modelFile, line)) {
            if (line == "\n") continue;
            w_.push_back(std::stod(line));
        }
    }

    void checkData(const Objects& o, const Vector& c) {
        if (o.empty()) {
            throw std::runtime_error("Training set is empty!");
        }
        if (o.size() != c.size()) {
            throw std::runtime_error("Mismatch between Objects and Classes");
        }
    }

    const Function& lossFuncionRaw(LossFunction lf) noexcept {
        static std::map<LossFunction, Function> data = {
                {LossFunction::V, V},
                {LossFunction::Q, Q},
                {LossFunction::Q3, Q3},
                {LossFunction::Q4, Q4},
                {LossFunction::S, S},
                {LossFunction::L, L},
                {LossFunction::E, E},
        };
        return data[lf];
    }

    const Function& lossFunctionDiff(LossFunction lf) noexcept {
        static std::map<LossFunction, Function> data = {
                {LossFunction::V, diffV},
                {LossFunction::Q, diffQ},
                {LossFunction::Q3, diffQ3},
                {LossFunction::Q4, diffQ4},
                {LossFunction::S, diffS},
                {LossFunction::L, diffL},
                {LossFunction::E, diffE},
        };
        return data[lf];
    }

    LossFunction lossFuncionByName(const std::string& name) {
        static std::map<std::string, LossFunction> data = {
                {"V", LossFunction::V},
                {"Q", LossFunction::Q},
                {"Q2", LossFunction::Q},
                {"Q3", LossFunction::Q3},
                {"Q4", LossFunction::Q4},
                {"S", LossFunction::S},
                {"L", LossFunction::L},
                {"E", LossFunction::E},
        };
        return data[name];
    }

    std::string lossFunctionToName(LossFunction lf) noexcept {
        std::map<LossFunction, std::string> data = {
                {LossFunction::V, "V"},
                {LossFunction::Q, "Q"},
                {LossFunction::Q, "Q2"},
                {LossFunction::Q3, "Q3"},
                {LossFunction::Q4, "Q4"},
                {LossFunction::S, "S"},
                {LossFunction::L, "L"},
                {LossFunction::E, "E"},
        };
        return  data[lf];
    }

    const KernelFunction& kernelRaw(Kernel k) noexcept {
        static std::map<Kernel, KernelFunction> data {
                {Kernel::Homogenous1, [](const Vector& a, const Vector& b) {
                    return dot(a, b);
                }},
                {Kernel::Homogenous3, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b), 3);
                }},
                {Kernel::Inhomogenius1, [](const Vector& a, const Vector& b) {
                    return dot(a, b) + 1;
                }},
                {Kernel::Inhomogenius3, [](const Vector& a, const Vector& b) {
                    return pow(dot(a, b) + 1, 3);
                }},
                {Kernel::Radial, [](const Vector& a, const Vector& b) {
                    return NAN;
                }},
                {Kernel::GaussianRadial, [](const Vector& a, const Vector& b) {
                    return NAN;
                }},
                {Kernel::Hyperbolic, [](const Vector& a, const Vector& b) {
                    return NAN;
                }}};
        return data[k];
    }

    Kernel kernelByName(const std::string& name) noexcept {
        return Kernel::Homogenous1;
    }

    std::string kernelToName(Kernel k) noexcept {
        return "";
    }
}

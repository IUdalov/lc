#include "bayes.h"

#include "debug.h"
#include <random>
#include <cmath>
#include <cassert>

namespace lc {

using namespace internal;

namespace {

double teta(double e, Distribution distribution) {
    switch(distribution) {
        case Distribution::Gauss:
            return e;
        case Distribution::Poisson:
            return log(e);
        case Distribution::Bernoulli:
        case Distribution::Binomial:
            return compare(e, 0) ? 0 : log(e / (1 - e));
        default:
            std::abort();
    }
}

double phi(double dis, Distribution distribution) {
    switch(distribution) {
        case Distribution::Gauss:
            return compare(dis, 0) ? 1 : dis;
        case Distribution::Poisson:
        case Distribution::Bernoulli:
        case Distribution::Binomial:
            return 1;
        default:
            std::abort();
    }
}

Vector realNaiveBayes(const Problem& p, Distribution distribution) {
    size_t features = p[0].size();
    size_t objects = p.entries().size();

    std::pair<Vector, Vector> mat;
    mat.first.assign(features, 0);
    mat.second.assign(features, 0);

    std::pair<Vector, Vector> disp;
    disp.first.assign(features, 0);
    disp.second.assign(features, 0);
    size_t firstClassCount = 0;

    for(const auto& entry : p.entries()) {
        if (entry.y() == 1) firstClassCount++;
        auto& m = entry.y() == 1 ? mat.first : mat.second;
        auto& d = entry.y() == 1 ? disp.first : disp.second;

        for(size_t i = 0; i < features; i++) {
            m[i] += entry[i];
            d[i] += entry[i] * entry[i];
        }
    }

    if (firstClassCount == 0 || firstClassCount == objects)
        throw std::runtime_error("Only one class is present");

    for(size_t i = 0; i < features; i++) {
        mat.first[i] = mat.first[i] / firstClassCount;
        disp.first[i] = disp.first[i] / firstClassCount;

        mat.second[i] = mat.second[i] / (objects - firstClassCount);
        disp.second[i] = disp.second[i] / (objects - firstClassCount);
    }

    for(size_t i = 0; i < features; i++) {
        disp.first[i] = disp.first[i] - mat.first[i] * mat.first[i];
        disp.second[i] = disp.second[i] - mat.second[i] * mat.second[i];
    }

    Vector w(features, 0);
    for(const auto& entry : p.entries()) {
        for(size_t i = 0; i < features; i++) {
            auto& m = entry.y() == 1 ? mat.first : mat.second;
            auto& d = entry.y() == 1 ? disp.first : disp.second;

            w[i] += entry.y() * teta(m[i], distribution) / phi(d[i], distribution);
        }
    }

    return w;
}

Vector randomVector(const Problem& p) {
    Vector v(p[0].size(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for(auto& elem : v) {
        elem = dis(gen);
    }
    return v;
}

} // namespace

namespace internal {

Vector naiveBayes(const Problem& p, Distribution distribution) {
    if (distribution == Distribution::Random) {
        return randomVector(p);
    } else {
        return realNaiveBayes(p, distribution);
    }
}

} } // namespace lc::interal
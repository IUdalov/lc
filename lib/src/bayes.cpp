#include "bayes.h"

#include "debug.h"
#include <cmath>
#include <cassert>

namespace lc {

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
            assert(false);
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
            assert(false);
    }
}

} // namespace

Vector naiveBayes(const Problem& p, Distribution distribution) {
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

} // namespace lc
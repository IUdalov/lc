#include "bayes.h"

#include "debug.h"
#include <cmath>

namespace lc {

Vector naiveBayes(const Problem& p) {
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

            if (compare(d[i], 0)) {
                d[i] = 1;
            }
            w[i] += entry.y() * m[i] / d[i];
        }
    }

    return w;
}

Vector oldNaiveBayes(const Problem& p) {
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

} // namespace lc
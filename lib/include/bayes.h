#pragma once

#include "data.h"
#include <map>
#include <string>

namespace lc {

enum class Distribution {
    Gauss,
    Poisson,
    Bernoulli,
    Binomial
};

Vector naiveBayes(const Problem& p, Distribution distribution = Distribution::Gauss);

inline Distribution distributionFromName(const std::string& name) {
    static std::map<std::string, Distribution> data = {
            {"Gauss", Distribution::Gauss},
            {"Normal", Distribution::Gauss},
            {"Poisson", Distribution::Poisson},
            {"Bernoulli", Distribution::Bernoulli},
            {"Binomial", Distribution::Binomial}
    };

    return data[name];
}

} // namespace lc
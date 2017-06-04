#pragma once

#include "data.h"
#include <map>
#include <string>

namespace lc {

enum class Distribution {
    Random,
    Gauss,
    Poisson,
    Bernoulli,
    Binomial
};

inline Distribution distributionFromName(const std::string& name) {
    static std::map<std::string, Distribution> data = {
            {"Random", Distribution::Random},
            {"Gauss", Distribution::Gauss},
            {"Normal", Distribution::Gauss},
            {"Poisson", Distribution::Poisson},
            {"Bernoulli", Distribution::Bernoulli},
            {"Binomial", Distribution::Binomial}
    };

    return data[name];
}

namespace internal {

Vector naiveBayes(const Problem& p, Distribution distribution = Distribution::Gauss);

} } // namespace lc::internal
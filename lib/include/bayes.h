#pragma once

#include "data.h"

namespace lc {

enum class Distribution {
    Gauss,
    Poisson,
    Bernoulli,
    Binomial
};

Vector naiveBayes(const Problem& p, Distribution distribution = Distribution::Gauss);

} // namespace lc
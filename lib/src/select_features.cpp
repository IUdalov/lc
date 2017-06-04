#include "select_features.h"

#include <set>
#include <cmath>

namespace lc {
namespace internal {

void selectFeatures(Vector& v, size_t useNFeatures) {
    std::set<double> maxs;

    for(const auto elem : v) {
        double abs = fabs(elem);
        if (maxs.size() < useNFeatures) {
            maxs.insert(abs);
        } else if (*maxs.begin() < abs) {
            maxs.erase(maxs.begin());
            maxs.insert(abs);
        }
    }
    double mean = *maxs.begin();

    for(auto& elem : v) {
        if (fabs(elem) < mean)
            elem = 0;
    }
}

} } // namespace lc::internal
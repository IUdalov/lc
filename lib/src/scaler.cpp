#include "scaler.h"

namespace lc {

Scaler::Scaler(const Problem& p) {
    size_t features = p[0].x().size();
    Vector maxs(features, -1000000000000000);
    Vector mins(features, 1000000000000000);

    for(auto& e : p.entries()) {
        for(size_t i = 0; i < features; i++) {
            double value = e.x()[i];
            if (value < mins[i]) mins[i] = value;
            if (value > maxs[i]) maxs[i] = value;
        }
    }

    factor_.assign(features, 1);
    offset_.assign(features, 0);

    for(size_t i = 0; i < features; i++) {
        if (compare(maxs[i], mins[i]))
            continue;

        factor_[i] = 1.0 / (maxs[i] - mins[i]);
        offset_[i] = (maxs[i] + mins[i]) / 2.0;
    }
}

void Scaler::apply(Vector& v) const {
    for(size_t i = 0; i < v.size(); i++)
        v[i] = (v[i] - offset_[i]) * factor_[i];
}

void Scaler::apply(Problem& p) const {
    for(auto& e : p.entries())
        apply(e.x());
}

} // namespace lc
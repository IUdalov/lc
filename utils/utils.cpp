#include "utils/utils.h"

#include "debug.h"

#include <sstream>
#include <regex>
#include <fstream>
#include <random>
#include <map>

namespace lc {

double checkData(const Model &model, const Problem& p) {
    size_t errors = 0;
    for (size_t i = 0; i < p.entries().size(); i++) {
        double res = model.predict(p[i].x());
        if (res * p[i].y() < 0) { errors++; }
    }

    return static_cast<double>(errors) / static_cast<double>(p.entries().size());
}

Problem generateNormalData(size_t objects,
                           size_t features,
                           double stddiv,
                           double offset,
                           const std::string &seed) {
    Problem res;
    std::seed_seq s(seed.begin(), seed.end());
    std::default_random_engine re(s);
    std::normal_distribution<double> nd(0, stddiv);


    for (size_t oCount = 0; oCount < objects; oCount++) {
        Vector tempFeatures(features);
        for (size_t fCount = 0; fCount < features; fCount++) {
            tempFeatures[fCount] = nd(re) + (oCount % 2 == 0 ? offset : -offset);
        }
        res.add(Entry(oCount % 2 == 0 ? 1 : -1, std::move(tempFeatures)));
    }
    return res;
}

} // namespace lc
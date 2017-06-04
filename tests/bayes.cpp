#include <boost/test/unit_test.hpp>

#include <bayes.h>
#include <utils/utils.h>

using namespace lc;
using namespace lc::internal;

BOOST_AUTO_TEST_CASE(bayes) {
    std::vector<size_t> testSpec {10, 50, 100};
    for(auto t : testSpec) {
        const auto problem = generateNormalData(t, t / 2, 1, 0.3,"BB King");

        const auto w = naiveBayes(problem);

        size_t errors = 0;
        for (size_t i = 0; i < problem.entries().size(); i++) {
            auto res = dot(problem[i].x(), w);
            if (res * problem[i].y() < 0) { errors++; }
        }

        double errRate = static_cast<double>(errors) / static_cast<double>(problem.entries().size());

        BOOST_CHECK(errRate < 0.3);
    }
}

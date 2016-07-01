#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(bayes) {
    std::vector<size_t> testData {10, 50, 100};
    for(auto t : testData) {
        Objects data;
        Vector classes;
        generateNormalData(data, classes, t, t / 2, 1, 0.3,"BB King");

        Model m;
        m.maximumStepsNumber(0);
        m.train(data, classes);

        double errors = checkData(m, data, classes);
        BOOST_CHECK(errors < 0.3);
    }
}

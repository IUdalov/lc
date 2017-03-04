#include <boost/test/unit_test.hpp>

#include <debug.h>
#include <lc.h>
#include <utils/utils.h>
#include <fstream>
#include <iostream>
#include <array>

using namespace lc;

BOOST_AUTO_TEST_CASE(precisionTest) {
    size_t objects = 10;
    size_t features = 10;

    auto p = generateNormalData(objects, features, 1, 0.3);
    std::vector<size_t > steps = {5,10,15,20, 40,80,100,120, 200, 300};

    double precisionBefore = std::numeric_limits<double>::max();
    for(auto ms : steps) {
        Model m;
        m.lossFunction(loss_functions::Q);
        m.classifier({1,4,32,64,512, 1,4,32,64,512});
        m.maximumStepsNumber(ms);
        m.c(0.01);
        auto i = m.train(p, true);
        double errors = checkData(m, p);
        BOOST_CHECK(i.precision < precisionBefore);
        precisionBefore = i.precision;
    }
}

BOOST_AUTO_TEST_CASE(converges) {
    size_t objects = 100;
    size_t features = 10;
    size_t testObjects = 10000;
    std::vector<size_t > steps = {5, 10, 15, 20, 40, 80, 100, 120, 200, 300};
    Vector initial({1, 4, -32, 64, -512, 1, 4, 32, -64, 512});

    auto train = generateNormalData(objects, features, 1, 0.3);
    auto test = generateNormalData(testObjects, features, 1, 0.3, "pony");

    Model base;
    base.classifier(initial);
    base.maximumStepsNumber(0);

    auto errBefore = checkData(base, test);
    double precisionBefore = std::numeric_limits<double>::max();
    for(auto ms : steps) {
        Model m;
        m.lossFunction(loss_functions::Q);
        m.classifier(initial);
        m.maximumStepsNumber(ms);
        m.c(0.01);

        auto i = m.train(train, true);
        double errors = checkData(m, test);
        DEBUG << "MS: " << ms << std::endl;
        BOOST_CHECK(errors <= errBefore);
        BOOST_CHECK(i.precision < precisionBefore);

        precisionBefore = i.precision;
        errBefore = errors;
    }
}


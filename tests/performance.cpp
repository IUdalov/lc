#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>
#include <fstream>
#include <iostream>

using namespace lc;

BOOST_AUTO_TEST_CASE(precisionTest) {
    size_t objects = 10;
    size_t features = 10;
    Objects data;
    Vector classes;

    generateNormalData(    data,     classes, objects, features, 1, 0.3);
    std::vector<size_t > steps = {5,10,15,20, 40,80,100,120, 200, 300};

    double precisionBefore = std::numeric_limits<double>::max();
    for(auto ms : steps) {
        Model m;
        m.lossFunction(LossFunction::Q);
        m.classifier({1,4,32,64,512, 1,4,32,64,512});
        m.maximumStepsNumber(ms);
        m.c(0.01);
        auto i = m.train(data, classes, true);
        double errors = checkData(m, data, classes);
        BOOST_CHECK(i.precision < precisionBefore);
        precisionBefore = i.precision;
    }
}

BOOST_AUTO_TEST_CASE(converges) {
    std::vector<LossFunction> lossFuncs({
        LossFunction::V,
        LossFunction::Q,
        LossFunction::Q3,
        LossFunction::Q4,
        LossFunction::L,
        LossFunction::S,
        LossFunction::E});

    size_t objects = 100;
    size_t features = 10;
    size_t testObjects = 10000;
    Objects data, testData;
    Vector classes, testClasses;
    std::vector<size_t > steps = {5,10,15,20, 40,80,100,120, 200, 300};
    Vector initial({1,4,32,64,512, 1,4,32,64,512});

    generateNormalData(    data,     classes, objects, features, 1, 0.3);
    generateNormalData(testData, testClasses, objects, features, 1, 0.3, "pony");

    Model base;
    base.classifier(initial);
    base.maximumStepsNumber(0);

    auto errBefore = checkData(base, data, classes);
    double precisionBefore = std::numeric_limits<double>::max();
    for(auto ms : steps) {
        Model m;
        m.lossFunction(LossFunction::Q);
        m.classifier(initial);
        m.maximumStepsNumber(ms);
        m.c(0.01);

        auto i = m.train(data, classes, true);
        double errors = checkData(m, testData, testClasses);

        BOOST_CHECK(errors <= errBefore);
        BOOST_CHECK(i.precision < precisionBefore);

        precisionBefore = i.precision;
        errBefore = errors;
    }
}

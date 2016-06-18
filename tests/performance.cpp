#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>
#include <fstream>
#include <iostream>

using namespace lc;


BOOST_AUTO_TEST_CASE(precisionTest) {
    size_t objects = 100;
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
        BOOST_CHECK(i.precision < precisionBefore);
        precisionBefore = i.precision;
    }
}

/*
BOOST_AUTO_TEST_CASE(currentTest) {
    std::vector<LossFunction> lossFuncs({
        LossFunction::V,
        LossFunction::Q,
        LossFunction::Q3,
        LossFunction::Q4,
        LossFunction::L,
        LossFunction::S,
        LossFunction::E});

    size_t objects = 10000;
    size_t features = 8000;
    size_t testObjects = 10000;
    Objects data, testData;
    Vector classes, testClasses;

    generateNormalData(    data,     classes, objects, features, 1, 0.3);
    generateNormalData(testData, testClasses, objects, features, 1, 0.3, "pony");

    Model base;
    base.classifier({1,4,32,64,512, 1,4,32,64,512});
    base.maximumStepsNumber(0);
    //base.train(data,classes);
    auto errBefore = 1 - checkData(base, data, classes);

    std::ofstream out("/Users/iudalov/Current/lc/build/tmp");
    out.precision(5);
    std::vector<size_t > steps = {5,10,15,20, 40,80,100,120, 200, 300};

    double precisionBefore = std::numeric_limits<double>::max();
    for(auto ms : steps) {
        Model m;
        m.lossFunction(LossFunction::Q);
        m.classifier({1,4,32,64,512, 1,4,32,64,512});
        m.maximumStepsNumber(ms);
        m.c(0.01);
        auto i = m.train(data, classes, true);
        BOOST_CHECK(i.precision < precisionBefore);
        precisionBefore = i.precision;
    }
}
*/


BOOST_AUTO_TEST_CASE(currentTest) {
    Objects o;
    Vector c;

    readCSVFile("/Users/iudalov/Current/lc/data/smsspamcollection/sms_train.csv", o, c);

    Model m;

    std::vector<size_t > steps = {1,2,3,4,5};
    for(auto ms : steps) {
        std::cout << "training " << ms << std::endl;
        Info i = m.train(o, c);
        std::cout << "done" << std::endl;
        std::cout << "Predicting..." << std::endl;
        double err = checkData(m, o, c);
        std::cout << "Assurance:" << err << std::endl;
    }
}
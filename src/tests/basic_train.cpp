#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

#include <map>
#include <iostream>
#include <fstream>

using namespace lc;

BOOST_AUTO_TEST_CASE(simpleTrainWithQ) {
    size_t objects = 10;
    size_t features = 3;
    size_t testObjects = 100;
    Objects data, testData;
    Vector classes, testClasses;

    generateNormalData(
            data,
            classes,
            objects,
            features,
            1, // stddiv
            0.5 // offset
    );

    generateNormalData(
            testData,
            testClasses,
            testObjects,
            features,
            1, // stddiv
            0.5, // offset
            "test seed"
    );

    Model model;
    model.setLossFunction(Q, diffQ);
    model.setData(data, classes);
    model.setC(0.01);
    model.setClassifier({1,2,3});
    double errorsBefore = checkData(model, testData, testClasses);
    auto info = model.train(true);
    double errorsAfter = checkData(model, testData, testClasses);
    (void)info;
    BOOST_CHECK(errorsBefore >= errorsAfter);
}


using std::pair;
typedef std::pair<double, double> dpair;
BOOST_AUTO_TEST_CASE(basicTrain) {
    std::map<char, pair<Function, Function>> lossFuncs({
            {'1', {V, diffV}},
            {'2', {Q, diffQ}},
            {'3', {Q3, diffQ3}},
            {'4', {Q4, diffQ4}},
            //{'L', {L, diffL}},
            //{'S', {S, diffS}},
            //{'/E', {E, diffE}},
    });

    // pairs of stddiv and offset
    std::vector<std::pair<double, double>> stddivAndOffsets({
            dpair(1,0.2),
            dpair(1,0.5),
            dpair(1,0.57),
    });

    size_t objects = 50;
    size_t testObjects = 200;
    size_t features = 5;

    Objects data, testData;
    Vector classes, testClasses;

    Vector wAbout(features);
    for(size_t fCount = 0; fCount < features; fCount++) {
        wAbout[fCount] = fCount * 3;
    }

    size_t spoiledClassifier = 0;
    std::vector<Info> stats;

    for(auto stddivAndOffset : stddivAndOffsets) {
        generateNormalData(
                data,
                classes,
                objects,
                features,
                stddivAndOffset.first, // stddiv
                stddivAndOffset.second // offset
        );

        generateNormalData(
                testData,
                testClasses,
                testObjects,
                features,
                stddivAndOffset.first, // stddiv
                stddivAndOffset.second, // offset
                "Test seed"
        );

        for(auto func : lossFuncs) {
            Model model;

            model.setLossFunction(
                    func.second.first, // just function
                    func.second.second // diff
            );

            model.setData(data, classes);
            model.setC(0.001/stddivAndOffset.second);
            model.setPrecision(0.0001);
            model.setMaximumStepsNumber(1000);
            model.setClassifier(wAbout);

            double errorsBefore = checkData(model, testData, testClasses);
            Info info = model.train(true);
            double errorsAfter = checkData(model, testData, testClasses);

            info.errorsBefore = errorsBefore;
            info.errorsAfter = errorsAfter;
            info.name = func.first;

            stats.push_back(info);


            if (errorsBefore < errorsAfter) {
                spoiledClassifier++;
            }
        }
    }
    BOOST_CHECK(spoiledClassifier <= (stats.size()/2));

    logInfoToFile(stats, "basic_train.log");
}
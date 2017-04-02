#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(simpleTrainWithQ) {
    size_t objects = 10;
    size_t features = 3;
    size_t testObjects = 100;

    auto train = generateNormalData(
            objects,
            features,
            1, // stddiv
            0.5 // offset
    );

    auto test = generateNormalData(
            testObjects,
            features,
            1, // stddiv
            0.5, // offset
            "test seed"
    );

    Model model;
    model.lossFunction(loss_functions::Q);
    model.c(0.01);
    model.precision(0.2);
    model.classifier({-1, -2, -3});
    double errorsBefore = checkData(model, test);
    auto info = model.train(train, true, true);
    double errorsAfter = checkData(model, test);
    (void)info;
    BOOST_CHECK_GE(errorsBefore, errorsAfter);
}

/*
typedef std::pair<double, double> dpair;

BOOST_AUTO_TEST_CASE(basicTrain) {
    std::vector<LossFunction> lossFuncs({LossFunction::V, LossFunction::Q, LossFunction::Q3, LossFunction::Q4});
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

            model.lossFunction(func);

            model.c(0.001/stddivAndOffset.second);
            model.precision(0.0001);
            model.maximumStepsNumber(1000);
            model.classifier(wAbout);

            double errorsBefore = checkData(model, testData, testClasses);
            Info info = model.train(data, classes, true);
            double errorsAfter = checkData(model, testData, testClasses);

            info.errorsBefore = errorsBefore;
            info.errorsAfter = errorsAfter;
            info.descr = lossFunctionToName(func);

            stats.push_back(info);

            if (errorsBefore < errorsAfter) {
                spoiledClassifier++;
            }
        }
    }
    BOOST_CHECK(spoiledClassifier <= (stats.size()/2));

    logInfoToFile(stats, "basic_train.log");
}
 */
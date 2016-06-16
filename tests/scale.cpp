#include <boost/test/unit_test.hpp>
#include <lc.h>

#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(scaleSample) {
    std::vector<Objects> samples;
    samples.push_back({
                              {1, 2, 5},
                              {5, 3, 1},
                              {1, 2, 4},
                              {0, 0, 0}
                      });
    samples.push_back({
                              { 0,  1,  3, -4},
                              {-1,  2,  4,  2},
                              { 0, -1, -2, -3},
                      });
    samples.push_back({
                              { 0.1,  3},
                              {-0.1,  2},
                              {-.01, -1},
                              {0.01, -3},
                      });
    for(auto& sample : samples) {
        Vector factor;
        Vector offset;
        Objects toScale = sample;
        scaleData(toScale, 1, factor, offset);

        for(size_t i = 0; i < sample.size(); i++) {
            Vector unscaled = toScale[i];
            unscaleVector(unscaled, factor, offset);
            for (size_t j = 0; j < sample[i].size(); j++) {
                BOOST_CHECK(isSame(sample[i][j], unscaled[j]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(trainWithScale) {
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
    auto info = model.train(data, classes);
    double errors = checkData(model, testData, testClasses);
    BOOST_CHECK(errors <= 2.3);
}

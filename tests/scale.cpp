#include <boost/test/unit_test.hpp>
#include <lc.h>

#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(scaleSample) {
    Problem p1;
    p1.add(Entry(0, {1, 2, 5}));
    p1.add(Entry(0, {5, 3, 1}));
    p1.add(Entry(0, {1, 2, 4}));
    p1.add(Entry(0, {0, 0, 0}));

    Problem p2;
    p2.add(Entry(0, { 0,  1,  3, -4}));
    p2.add(Entry(0, {-1,  2,  4,  2}));
    p2.add(Entry(0, { 0, -1, -2, -3}));

    Problem p3;
    p3.add(Entry(0, { 0.1,  3}));
    p3.add(Entry(0, {-0.1,  2}));
    p3.add(Entry(0, {-.01, -1}));
    p3.add(Entry(0, {0.01, -3}));

    std::vector<Problem> samples;
    samples.emplace_back(std::move(p1));
    samples.emplace_back(std::move(p2));
    samples.emplace_back(std::move(p3));

    for(auto& sample : samples) {
        Vector factor;
        Vector offset;
        auto toScale = sample.dup();
        scaleData(toScale, 1, factor, offset);

        for(size_t i = 0; i < sample.entries().size(); i++) {
            Vector unscaled = toScale[i].x();
            unscaleVector(unscaled, factor, offset);
            for (size_t j = 0; j < sample[i].size(); j++) {
                BOOST_CHECK(compare(sample[i][j], unscaled[j]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(trainWithScale) {
    size_t objects = 100;
    size_t features = 3;
    size_t testObjects = 1000;

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
    model.train(train);
    double errors = checkData(model, test);
    BOOST_CHECK(errors <= 2.3);
}

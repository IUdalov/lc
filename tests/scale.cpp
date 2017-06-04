#include <boost/test/unit_test.hpp>
#include <lc.h>

#include <utils/utils.h>

using namespace lc;
using namespace lc::internal;

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
    p3.add(Entry(0, { 0.1,  0.3}));
    p3.add(Entry(0, {-0.1,  0.2}));
    p3.add(Entry(0, {-.01, -0.3}));
    p3.add(Entry(0, {0.01, -0.2}));

    std::vector<Problem> samples;
    samples.emplace_back(std::move(p1));
    samples.emplace_back(std::move(p2));
    samples.emplace_back(std::move(p3));

    for(auto& sample : samples) {
        Scaler r(sample);
        r.apply(sample);

        for(auto& e : sample.entries())
            for(auto& elem : e.x()) {
                BOOST_CHECK_LE(e, 1);
                BOOST_CHECK_GE(e, -1);
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

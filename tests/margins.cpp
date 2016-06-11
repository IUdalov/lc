#include <boost/test/unit_test.hpp>

#include <lc.h>
#include "utils/utils.h"

using namespace lc;

BOOST_AUTO_TEST_CASE(margins) {
    Model model;
    Objects data({ {-1, 1}, {7, 5}, {-0.1, 0}, {0.1, 0} });
    Vector classes({1, -1, 1, -1});

    model.setData(data, classes);
    model.setC(1);

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with Q");
    model.setLossFunction(Q, diffQ);

    model.setClassifier({-1, 1});
    model.toMargins();

    BOOST_CHECK(about(2, model.getMargins()[0]));
    BOOST_CHECK(about(2, model.getMargins()[1]));

    model.toClassifier();

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with L");
    model.setLossFunction(V, diffV);

    model.setClassifier({-1, 1});
    model.toMargins();

    BOOST_CHECK(about(2, model.getMargins()[0]));
    BOOST_CHECK(about(2, model.getMargins()[1]));

    model.toClassifier();

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with L");
    model.setLossFunction(L, diffL);

    model.setClassifier({-1, 1});
    model.toMargins();

    BOOST_CHECK(about(2, model.getMargins()[0]));
    BOOST_CHECK(about(2, model.getMargins()[1]));

    model.toClassifier();

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }
}

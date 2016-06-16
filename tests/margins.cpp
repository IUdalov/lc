#include <boost/test/unit_test.hpp>

#include <lc.h>
#include "utils/utils.h"

using namespace lc;

BOOST_AUTO_TEST_CASE(margins) {
    Model model;
    Objects data({ {-1, 1}, {7, 5}, {-0.1, 0}, {0.1, 0} });
    Vector classes({1, -1, 1, -1});

    model.c(1);

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with Q");
    model.lossFunction(LossFunction::Q);

    model.classifier({-1, 1});
    model.toMargins(data, classes);

    BOOST_CHECK(about(2, model.margins()[0]));
    BOOST_CHECK(about(2, model.margins()[1]));

    model.toClassifier(data, classes);

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with L");
    model.lossFunction(LossFunction::V);

    model.classifier({-1, 1});
    model.toMargins(data, classes);

    BOOST_CHECK(about(2, model.margins()[0]));
    BOOST_CHECK(about(2, model.margins()[1]));

    model.toClassifier(data, classes);

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with L");
    model.lossFunction(LossFunction::L);

    model.classifier({-1, 1});
    model.toMargins(data, classes);

    BOOST_CHECK(about(2, model.margins()[0]));
    BOOST_CHECK(about(2, model.margins()[1]));

    model.toClassifier(data, classes);

    for(size_t i = 0; i < data.size(); i++) {
        BOOST_CHECK(model.predict(data[i]) == classes[i]);
    }
}

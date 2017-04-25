#include <boost/test/unit_test.hpp>

#include <lc.h>
#include "utils/utils.h"

using namespace lc;

BOOST_AUTO_TEST_CASE(margins) {
    Model model;
    Problem p;
    p.add(Entry(1, {-1, 1}));
    p.add(Entry(-1, {7, 5}));
    p.add(Entry(1, {-0.1, 0}));
    p.add(Entry(-1, {0.1, 0}));

    model.c(1);

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with Q");
    model.lossFunction(loss_functions::X2);

    model.classifier({-1, 1});
    model.toMargins(p);

    BOOST_CHECK(compare(2, model.margins()[0]));
    BOOST_CHECK(compare(2, model.margins()[1]));

    model.toClassifier(p);

    for(size_t i = 0; i < p.entries().size(); i++) {
        BOOST_CHECK_GT(model.predict(p[i].x()) * p[i].y(), 0);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with V");
    model.lossFunction(loss_functions::X);

    model.classifier({-1, 1});
    model.toMargins(p);

    BOOST_CHECK(compare(2, model.margins()[0]));
    BOOST_CHECK(compare(2, model.margins()[1]));

    model.toClassifier(p);

    for(size_t i = 0; i < p.entries().size(); i++) {
        BOOST_CHECK_GT(model.predict(p[i].x()) * p[i].y(), 0);
    }

    // ------------------------------------------------------------------------
    BOOST_TEST_MESSAGE("Testing margins with L");
    model.lossFunction(loss_functions::L);

    model.classifier({-1, 1});
    model.toMargins(p);

    BOOST_CHECK(compare(2, model.margins()[0]));
    BOOST_CHECK(compare(2, model.margins()[1]));

    model.toClassifier(p);

    for(size_t i = 0; i < p.entries().size(); i++) {
        BOOST_CHECK_GT(model.predict(p[i].x()) * p[i].y(), 0);
    }
}

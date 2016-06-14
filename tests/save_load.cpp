#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(saveLoad) {
    Model model;

    model.setClassifier({3,2,1});
    model.save("saveLoad.lc");
    model.load("saveLoad.lc");

    auto w = model.getClassifier();
    BOOST_CHECK(about(w[0], 3));
    BOOST_CHECK(about(w[1], 2));
    BOOST_CHECK(about(w[2], 1));
}
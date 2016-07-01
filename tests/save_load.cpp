#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(saveLoad) {
    Model model;

    model.classifier({3, 2, 1});
    model.save("saveLoad.lc");
    model.load("saveLoad.lc");

    auto w = model.classifier();
    BOOST_CHECK(compare(w[0], 3));
    BOOST_CHECK(compare(w[1], 2));
    BOOST_CHECK(compare(w[2], 1));
}
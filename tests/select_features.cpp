#include <boost/test/unit_test.hpp>

#include "select_features.h"

using namespace lc;
using namespace lc::internal;

BOOST_AUTO_TEST_CASE(select3Features) {
    Vector v = {0, 1, 2, -10, 3, -1};
    selectFeatures(v, 3);

    BOOST_CHECK_EQUAL(v[0], 0);
    BOOST_CHECK_EQUAL(v[1], 0);
    BOOST_CHECK_EQUAL(v[2], 2);
    BOOST_CHECK_EQUAL(v[3], -10);
    BOOST_CHECK_EQUAL(v[4], 3);
    BOOST_CHECK_EQUAL(v[5], 0);
}

#include <boost/test/unit_test.hpp>
#include <lc.h>

#include <math.h>

using namespace lc;
using namespace lc::internal;

BOOST_AUTO_TEST_CASE(dotTest) {
    BOOST_CHECK(compare(dot(Vector({4,3}), Vector({5,1})), 4*5 + 3*1));
    BOOST_CHECK(compare(dot(Vector({1,1,1}), Vector({1,1,1})), 3));
    BOOST_CHECK(compare(dot(Vector({1,2,3,4}), Vector({4,3,2,1})), 4 + 6 + 6 + 4));
}

BOOST_AUTO_TEST_CASE(lengthTest) {
    BOOST_CHECK(compare(length(Vector({3,4})), 5));
    BOOST_CHECK(compare(length(Vector({1,2,3})), sqrt(1 + 4 + 9)));
}

BOOST_AUTO_TEST_CASE(distanceTest) {
    BOOST_CHECK(compare(distance(Vector({0,3}), Vector({0,0})), 3));
    BOOST_CHECK(compare(distance(Vector({3,0}), Vector({0,4})), 5));
    BOOST_CHECK(compare(distance(Vector({3,4}), Vector({4,3})), sqrt(2)));
}

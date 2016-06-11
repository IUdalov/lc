#include <boost/test/unit_test.hpp>
#include <lc.h>
#include <math.h>

const double SMALL_VALUE = 0.00000000000001;

BOOST_AUTO_TEST_CASE(dotFunction) {
    BOOST_CHECK(
        fabs(lc::dot(lc::Vector({4,3}), lc::Vector({5,1})) - (4*5 + 3*1)) < SMALL_VALUE
    );
    BOOST_CHECK(
        fabs(lc::dot(lc::Vector({1,1,1}), lc::Vector({1,1,1})) - 3) < SMALL_VALUE
    );
    BOOST_CHECK(
        fabs(lc::dot(lc::Vector({1,2,3,4}), lc::Vector({4,3,2,1})) - (4 + 6 + 6 + 4)) < SMALL_VALUE
    );
}

BOOST_AUTO_TEST_CASE(lengthFunction) {
    BOOST_CHECK(
        fabs(lc::length(lc::Vector({3,4})) - 5) < SMALL_VALUE
    );
    BOOST_CHECK(
        fabs(lc::length(lc::Vector({1,2,3})) - sqrt(1 + 4 + 9)) < SMALL_VALUE
    );
}

BOOST_AUTO_TEST_CASE(distanceFunction) {
    BOOST_CHECK(
        fabs(lc::distance(lc::Vector({0,3}), lc::Vector({0,0})) - 3) < SMALL_VALUE
    );
    BOOST_CHECK(
        fabs(lc::distance(lc::Vector({3,0}), lc::Vector({0,4})) - 5) < SMALL_VALUE
    );
    BOOST_CHECK(
        fabs(lc::distance(lc::Vector({3,4}), lc::Vector({4,3})) - sqrt(2)) < SMALL_VALUE
    );
}

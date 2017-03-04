#include <boost/test/unit_test.hpp>
#include <lc.h>

#include <math.h>

BOOST_AUTO_TEST_CASE(dot) {
    BOOST_CHECK(lc::compare(lc::dot(lc::Vector({4,3}), lc::Vector({5,1})), 4*5 + 3*1));
    BOOST_CHECK(lc::compare(lc::dot(lc::Vector({1,1,1}), lc::Vector({1,1,1})), 3));
    BOOST_CHECK(lc::compare(lc::dot(lc::Vector({1,2,3,4}), lc::Vector({4,3,2,1})), 4 + 6 + 6 + 4));
}

BOOST_AUTO_TEST_CASE(length) {
    BOOST_CHECK(lc::compare(lc::length(lc::Vector({3,4})), 5));
    BOOST_CHECK(lc::compare(lc::length(lc::Vector({1,2,3})), sqrt(1 + 4 + 9)));
}

BOOST_AUTO_TEST_CASE(distance) {
    BOOST_CHECK(lc::compare(lc::distance(lc::Vector({0,3}), lc::Vector({0,0})), 3));
    BOOST_CHECK(lc::compare(lc::distance(lc::Vector({3,0}), lc::Vector({0,4})), 5));
    BOOST_CHECK(lc::compare(lc::distance(lc::Vector({3,4}), lc::Vector({4,3})), sqrt(2)));
}

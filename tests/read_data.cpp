#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(readWriteData) {
    std::vector<Entry> src;
    src.emplace_back(Entry(1, {1, 2, 3}));
    src.emplace_back(Entry(-1, {0, 0, 9}));
    src.emplace_back(Entry(1, {-1, 0, -1}));
    src.emplace_back(Entry(-1, {1, 0, 0}));

    std::stringstream ss;
    for(const auto& e : src)
        ss << e;

    Problem dst = readProblem(ss);
    BOOST_CHECK_EQUAL(src.size(), dst.entries().size());
    for(size_t i = 0; i < src.size(); i++)
        BOOST_CHECK_EQUAL(src[i], dst[i]);
}
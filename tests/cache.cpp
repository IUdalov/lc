#include <boost/test/unit_test.hpp>
#include <lc.h>
#include <internal/cache.h>

#include <stdlib.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(cache) {
    Objects o({
            {1,2,3,4,5},
            {0,1,2,3,4},
            {3,4,5,6,7},
            {7,8,9,0,1},
            {3,3,2,3,5},
            {9,8,7,8,1},
            {3,4,5,6,7}}); // 7x5
    Vector c({1,1,-1,1});
    Cache cache(o, c, Kernel::Homogenous1);
    KernelFunction kf = kernelRaw(Kernel::Homogenous1);
    for (size_t m = 0; m < 100; m++) {
        size_t i = rand() % 7;
        size_t k = rand() % 7;
        compare(cache.at(i,k), c[i]*c[k]*kf(o[i],o[k]));
    };
}
#define BOOST_TEST_NO_MAIN
#include <boost/test/included/unit_test.hpp>

boost::unit_test::test_suite* init_unit_test(int, char **) {
    return 0;
}

int main(int argc, char** argv) {
    return boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
}

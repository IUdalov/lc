#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <sstream>

using namespace lc;


namespace  {

struct Fixture {
    Fixture() {
        problem.add(Entry( 1, { 3, 4,  5}));
        problem.add(Entry( 1, { 0,  0,  0}));
        problem.add(Entry(-1, {-1, -2, -3}));
    }

    Problem problem;
    std::stringstream buffer;
};

} // namespace

BOOST_FIXTURE_TEST_CASE(serializeResizer, Fixture) {
    Scaler origin(problem);
    buffer << origin;

    Scaler copy;
    buffer >> copy;

    for(const auto& e : problem.entries()) {
        Vector v1 = e.x();
        Vector v2 = e.x();

        origin.apply(v1);
        copy.apply(v2);

        for(size_t i = 0; i < v1.size(); i++) {
            BOOST_CHECK(compareWith(v1[i], v2[i], 0.0001));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(serializeModel, Fixture) {
    Model origin;
    origin.train(problem);
    buffer << origin;

    Model copy;
    buffer >> copy;

    for(const auto& e : problem.entries()) {
        double prob1 = origin.predict(e.x());
        double prob2 = copy.predict(e.x());
        BOOST_CHECK(compareWith(prob1, prob2, 0.00001));
    }
}
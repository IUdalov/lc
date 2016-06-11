// TODO: fix or delete
#ifdef DISABLED_TESTS

#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(easy3DataSet) {
    std::string path(dataPath(DataSets::easy3));

    Model model;
    Objects data;
    Vector classes;

    readCSVFile(path, data, classes);

    model.setLossFunction(Q, diffQ);
    model.setData(data, classes);
    model.setC(0.1);

    model.setClassifier({1, 1});
    double errorsNaive = checkData(model, data, classes);

    model.train(true);
    double errors = checkData(model, data, classes);

    BOOST_CHECK(errorsNaive > errors);
}

#endif
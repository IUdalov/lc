// TODO: fix or delete
#ifdef DISABLED_TESTS

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <lc.h>
#include <utils/utils.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(easyDataSet) {
    std::string path(dataPath(DataSets::easy));

    Model model;
    Objects data;
    Vector classes;

    readCSVFile(path, data, classes);

    model.setData(data, classes);
    model.setLossFunction(V, diffV);
    model.setC(1);

    model.train();

    double errors = checkData(model, data, classes);

    BOOST_CHECK(errors < 0.3);
}

#endif
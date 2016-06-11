// TODO: fix bayes
#ifdef DISABLED_TESTS

#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>

#include <iostream>

using namespace lc;

BOOST_AUTO_TEST_CASE(bayes) {
    std::vector<std::string> dataSets = csvDatasets();

    for(auto it = dataSets.begin(); it != dataSets.end(); it++) {
        std::string path = *it;
        Model model;
        Objects data;
        Vector classes;

        readCSVFile(path, data, classes);
        addDim(data);

        model.setC(1);
        model.setData(data, classes);

        model.bayes();

        double errors = checkData(model, data, classes);
        BOOST_CHECK_MESSAGE(errors < 0.3, "Bayes failed on " + path + ". Error: " + std::to_string(errors));
    }
}

#endif
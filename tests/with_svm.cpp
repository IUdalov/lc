#include <boost/test/unit_test.hpp>

#include <lc.h>
#include <utils/utils.h>
// #include <svm.h>

using namespace lc;

BOOST_AUTO_TEST_CASE(compateWithSVM) {
    Objects o;
    Vector c;
    readSVMFile("/Users/iudalov/Current/lc/third-party/libsvm/heart_scale", o, c);
    //writeSVMFile("/Users/iudalov/Current/lc/third-party/libsvm/heart_scale.plain", o, c);

    Model model;
    model.lossFunction(LossFunction::Q);
    model.setC(0.1);
    model.bayes(o, c);
    double errorsBefore = checkData(model, o, c);
    auto info = model.train(o, c);
    double errorsAfter = checkData(model, o, c);
    (void)info;
}

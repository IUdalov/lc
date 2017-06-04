
#include <lc.h>
#include <utils/utils.h>
#include <iostream>
#include <fstream>
#include <map>

#include <cstdlib>

using namespace lc;

namespace {

std::string envLookup(const std::string& name, const std::string& def) {
    char* res = getenv(name.c_str());
    return res == nullptr ? def : std::string(res);
}

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " training_file model_file" << std::endl;
    std::cout << "Environment variables:" << std::endl;
    std::cout << "\tLC_LF - loss function" << std::endl;
    std::cout << "\tLC_KERNEL - kernel for kernel trick" << std::endl;
    std::cout << "\tLC_C - algorithm parameter (> 0)" << std::endl;
    std::cout << "\tLC_STEPS - number of iterations" << std::endl;
    std::cout << "\tLC_INIT - type of initial approximation" << std::endl;
    std::cout << "\tLC_USE_N_FEATURES - number of features to use in initial approximation" << std::endl;
    std::cout << std::endl;

    std::cout << "Functions:" << std::endl;
    std::cout << "\tV(x)  = max(1-x, 0)" << std::endl;
    std::cout << "\tQ(x)  = {(1-x)^2, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ3(x) = {(1-x)^3, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ4(x) = {(1-x)^4, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tL(x)  = log2(1 + e^(-x))" << std::endl;
    std::cout << "\tS(x)  = 2/(1 + e^x)" << std::endl;
    std::cout << "\tE(x)  = e^x" << std::endl;
    std::cout << std::endl;

    std::cout << "Kernels:" << std::endl;
    std::cout << "\tH1, H2, H3" << std::endl;
    std::cout << "\tI1, I2, I3" << std::endl;
    std::cout << "\tHyperbolic, Radial, GaussianRadial" << std::endl;
    std::cout << std::endl;

    std::cout << "Initial approximations:" << std::endl;
    std::cout << "\tGauss" << std::endl;
    std::cout << "\tPoisson" << std::endl;
    std::cout << "\tBernoulli" << std::endl;
    std::cout << "\tRandom" << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string trainingFile(argv[1]);
    std::string modelFile(argv[2]);

    std::ifstream in(trainingFile);
    Problem p;
    in >> p;

    Model m;
    m.lossFunction(loss_functions::fromName(envLookup("LC_LF", "Q")));
    m.kernel(kernels::fromName(envLookup("LC_KERNEL", "H1")));
    m.c(std::stod(envLookup("LC_C", "1")));
    m.maximumStepsNumber(std::stoul(envLookup("LC_STEPS", "100")));
    m.approximation(distributionFromName(envLookup("LC_INIT", "Gauss")));
    m.precision(std::stod(envLookup("LC_PRECISION", "0.00000000001")));
    m.useNFeatures(std::stoul(envLookup("LC_USE_N_FEATURES", "0")));
    m.train(p);
    std::ofstream out(modelFile);
    out << m;

    m.log(std::cout);

    return 0;
}

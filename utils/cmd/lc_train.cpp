
#include <lc.h>
#include <utils/utils.h>
#include <iostream>
#include <fstream>
#include <map>

using namespace lc;

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " function kernel c max_steps training_file model_file" << std::endl;
    std::cout << "Functions:" << std::endl;
    std::cout << "\tV(x)  = max(1-x, 0)" << std::endl;
    std::cout << "\tQ(x)  = {(1-x)^2, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ3(x) = {(1-x)^3, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ4(x) = {(1-x)^4, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tL(x)  = log2(1 + e^(-x))" << std::endl;
    std::cout << "\tS(x)  = 2/(1 + e^x)" << std::endl;
    std::cout << "\tE(x)  = e^x" << std::endl;
    std::cout << "Kernels:" << std::endl;
    std::cout << "\tH1, H2, H3" << std::endl;
    std::cout << "\tI1, I2, I3" << std::endl;
    std::cout << "\tHyperbolic, Radial, GaussianRadial" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printUsage(argv[0]);
        return 1;
    }

    std::string lf(argv[1]);
    std::string kernel(argv[2]);
    std::string c(argv[3]);
    std::string rawSteps(argv[4]);
    std::string trainingFile(argv[5]);
    std::string modelFile(argv[6]);

    std::ifstream in(trainingFile);
    Problem p;
    in >> p;

    Model m;
    m.lossFunction(loss_functions::fromName(lf));
    m.kernel(kernels::fromName(kernel));
    m.c(std::stod(c));
    m.maximumStepsNumber(std::stoul(rawSteps));

    m.train(p);
    std::ofstream out(modelFile);
    out << m;

    m.log(std::cout);

    return 0;
}



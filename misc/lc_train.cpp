#include <iostream>
#include <map>
#include <lc.h>
#include <utils/utils.h>

using namespace lc;

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " [options] [function] c max_steps training_file model_file" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t-svm - read input file in SVM format" << std::endl;
    std::cout << "\t-csv - read input file in CSV format." << std::endl;
    std::cout << "Functions:" << std::endl;
    std::cout << "\tV(x)  = max(1-x, 0)" << std::endl;
    std::cout << "\tQ(x)  = {(1-x)^2, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ3(x) = {(1-x)^3, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tQ4(x) = {(1-x)^4, x < 1; 0, x >= 1}" << std::endl;
    std::cout << "\tL(x)  = log2(1 + e^(-x))" << std::endl;
    std::cout << "\tS(x)  = 2/(1 + e^x)" << std::endl;
    std::cout << "\tE(x)  = e^x" << std::endl;
}

const std::map<std::string, std::pair<Function, Function>> lossFuncs({
        {"V", {V, diffV}},
        {"Q", {Q, diffQ}},
        {"Q3", {Q3, diffQ3}},
        {"Q4", {Q4, diffQ4}},
        {"L", {L, diffL}},
        {"S", {S, diffS}},
        {"E", {E, diffE}},});

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFormat(argv[1]);
    std::string func(argv[2]);
    std::string rawC(argv[3]);
    std::string rawSteps(argv[4]);
    std::string trainingFile(argv[5]);
    std::string modelFile(argv[6]);
    Objects objects;
    Vector classes;
    Model m;

    if (inputFormat == "-svm") {
        readSVMFile(trainingFile, objects, classes);
    } else if (inputFormat == "-csv") {
        readCSVFile(trainingFile, objects, classes);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    if (lossFuncs.find(func) == lossFuncs.end()) {
        std::cerr << "Unable to locate function " << func << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    m.lossFunction(lossFuncionByName(func));
    m.setC(std::stod(rawC));
    m.maximumStepsNumber(std::stoll(rawSteps));

    Info i = m.train(objects, classes);
    m.save(modelFile);

    std::cout << "Info" << std::endl;
    std::cout << "objects: " << i.objects << std::endl;
    std::cout << "features: " << i.features << std::endl;
    std::cout << "c: " << i.c << std::endl;
    std::cout << "steps: " << i.steps << std::endl;
    std::cout << "precision: " << i.precision << std::endl;
    std::cout << "w: "; for(auto it : i.w) { std::cout << it << ", "; } std::cout << std::endl;
    std::cout << "old: "; for(auto it : i.w) { std::cout << it << ", "; } std::cout << std::endl;

    return 0;
}
#include <iostream>
#include <lc.h>
#include <utils/utils.h>

using namespace lc;

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " [options] test_file model_file" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t-svm - read input file in SVM format" << std::endl;
    std::cout << "\t-csv - read input file in CSV format." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    Objects objects;
    Vector classes;
    std::string inputFormat(argv[1]);

    if (inputFormat == "-svm") {
        readSVMFile(argv[2], objects, classes);
    } else if (inputFormat == "-csv") {
        readCSVFile(argv[2], objects, classes);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    Model m;
    m.load(argv[3]);


    double errors =checkData(m, objects, classes);
    std::cout << "Assurance: " << (1 - errors) << std::endl;
    return 0;
}
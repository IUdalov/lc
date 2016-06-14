#include <iostream>
#include <lc.h>
#include <utils/utils.h>

using namespace lc;

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " [options] objects features offset output_file" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t-svm - read input file in SVM format" << std::endl;
    std::cout << "\t-csv - read input file in CSV format." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input(argv[1]);
    int objects = std::stoul(argv[2]);
    int features = std::stoul(argv[3]);
    double offset = std::stod(argv[4]);
    std::string outputFile(argv[5]);

    Objects o;
    Vector c;

    generateNormalData(
            o,
            c,
            objects,
            features,
            1, // Standard deviation.
            offset, // Offset for each feature. offset * class
            outputFile); // just seed

    if (input == "-svm") {
        writeSVMFile(outputFile, o, c);
    } else if (input == "-csv") {
        // TODO: unsupported
        std::cerr << "TODO: Unsupported yet%(" << std::endl;
        // writeCSVFile()
    } else {
        printUsage(argv[0]);
        return 1;
    }
    return 0;
}

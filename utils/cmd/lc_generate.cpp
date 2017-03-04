#include <lc.h>
#include <utils/utils.h>

#include <iostream>
#include <fstream>

using namespace lc;

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << "  nobjects nfeatures offset output_file" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printUsage(argv[0]);
        return 1;
    }

    size_t objects = std::stoul(argv[1]);
    size_t features = std::stoul(argv[2]);
    double offset = std::stod(argv[3]);
    std::string outputFile(argv[4]);

    auto p = generateNormalData(
            objects,
            features,
            1, // Standard deviation.
            offset, // Offset for each feature. offset * class
            outputFile); // just seed

    std::ofstream oo;
    oo.open(outputFile);
    for(const auto& e : p.entries())
        oo << e;

    return 0;
}

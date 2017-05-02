#include <lc.h>
#include <utils/utils.h>

#include <iostream>
#include <fstream>

using namespace lc;

namespace {

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " <test_file> <model_file> <out_labels>" << std::endl;
}

template<typename T>
void printList(std::ostream& oo, const T& values) {
    for(const auto& value : values)
        oo << value << " ";
    oo << std::endl;
}

}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input =  argv[1];
    std::string model = argv[2];
    std::string output = argv[3];
    auto p = input == "stdin" ? readProblem(std::cin): readProblem(input);

    Model m;
    std::ifstream in(model);
    in >> m;

    std::vector<int> classes(p.entries().size());
    Vector predicted(p.entries().size());
    size_t errors = 0;

    for(size_t i = 0; i < p.entries().size(); i++) {
        predicted[i] = m.predict(p[i].x());
        if (predicted[i] * p[i].y() < 0) {errors++;}
        classes[i] = p[i].y();
    }

    if (output == "stdout") {
        printList(std::cout, classes);
        printList(std::cout, predicted);
    } else {
        std::ofstream oo;
        oo.open(output);
        printList(oo, classes);
        printList(oo, predicted);
    }

    std::cout << "Assurance: " << (1 - static_cast<double>(errors)/ static_cast<double>(predicted.size())) << std::endl;
    return 0;
}
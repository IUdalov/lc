#include <iostream>
#include <lc.h>
#include <utils/utils.h>

using namespace lc;

namespace {

void printUsage(const std::string progName) {
    std::cout << "Usage: " << progName << " test_file model_file" << std::endl;
}

template<typename T>
void printList(const std::string& name, const T& values) {
    std::cout << name << ": ";
    for(const auto& value : values)
        std::cout << value << " ";
    std::cout << std::endl;
}

}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    auto p = readProblem(argv[1]);

    Model m;
    m.load(argv[2]);

    std::vector<int> classes(p.entries().size());
    Vector predicted(p.entries().size());
    size_t errors = 0;

    for(size_t i = 0; i < p.entries().size(); i++) {
        predicted[i] = m.predict(p[i].x());
        if (predicted[i] != p[i].y()) {errors++;}
        classes[i] = p[i].y();
    }


    printList("Real", classes);
    printList("Predicted", predicted);

    std::cout << "Assurance: " << (1 - static_cast<double>(errors)/ static_cast<double>(predicted.size())) << std::endl;
    return 0;
}
#include <lc.h>

#include <fstream>

namespace lc {
void Model::save(const std::string& path) {
    if (w_.empty()) {
        throw std::runtime_error("Model was not configured");
    }
    std::ofstream out;
    out.open(path);

    for(auto it : w_) {
        out << it << std::endl;
    }

    out.close();
}

void Model::load(const std::string& path) {
    w_.clear();
    std::ifstream modelFile(path);
    if (!modelFile) {
        throw std::runtime_error("File " + path + " not found!");
    }
    std::string line;
    while(std::getline(modelFile, line)) {
        if (line == "\n") continue;
        w_.push_back(std::stod(line));
    }
}

}
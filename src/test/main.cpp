#include <math.h>
#include <iostream>

#include <lc.h>
#include <debug.h>

#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path to file>" << std::endl;
        exit(1);
    }
    std::cout << "Version: " << lc::getVersion() << std::endl;
    lc::LossFunction Q = [](float m){ return pow(m - 1, 2); };
    lc::LossFunction V = [](float m){ return std::max(1 - m, static_cast<float>(0.0)); };

    lc::Model model(Q);
    lc::Problem data;
    readCSVFile(argv[1], data);
    model.train(data);

    size_t errors = 0;
    for(auto it : data) {
        int res = model.predict(it.first);
        if (res != it.second) {errors++;}
        std::cout << "Pred clas: " << res << "; Real class:" << it.second << std::endl;
    }

    std::cout << "Totoal errors: " << 100 * static_cast<float>(errors)/static_cast<float>(data.size()) << "%" << std::endl; 
    return 0;
}


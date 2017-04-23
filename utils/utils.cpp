#include "utils/utils.h"

#include "debug.h"

#include <sstream>
#include <regex>
#include <fstream>
#include <random>
#include <map>

namespace lc {

LossFunction lossFunctionByName(const std::string& name) {
    static const std::map<std::string, LossFunction> nameToLF = {
            {"V",  loss_functions::V},
            {"Q",  loss_functions::Q},
            {"Q3", loss_functions::Q3},
            {"Q4", loss_functions::Q4},
            {"L",  loss_functions::L},
            {"S",  loss_functions::S},
            {"E",  loss_functions::E}};
    auto it = nameToLF.find(name);
    if (it == std::end(nameToLF))
        throw std::runtime_error("No such function " + name);
    return it->second;
}

Problem readProblem(std::istream& content) {
    if (!content)
        throw std::runtime_error("malformed ifstream");

    std::string line;
    Problem problem;
    std::regex tokenRegex("[e0-9\\+\\-\\.:]+");

    while (std::getline(content, line)) {
        auto token = std::sregex_iterator(line.begin(), line.end(), tokenRegex);
        auto end = std::sregex_iterator();

        int y = 0;
        auto yStr = token->str(); token++;
        if (yStr == "+1")  y = 1;
        else if (yStr == "-1") y = -1;
        else throw std::runtime_error("Unexpected token: " + yStr);

        std::vector<double> x;
        for(;token != end; token++) {
            auto vStr = token->str();
            size_t pos = vStr.find(":");
            if (pos == std::string::npos) throw std::runtime_error("Unexpected value: " + vStr);

            size_t ind = std::stoul(vStr.substr(0, pos));
            double val = std::stod(vStr.substr(pos + 1));

            if (val != val)
                throw std::runtime_error("NaN");

            x.resize(ind, 0);
            x[ind - 1] = val;
        }

        problem.add(Entry(y, x));
    }

    size_t max = 0;
    for(const auto& e : problem.entries())
        max = std::max(e.size(), max);
    for(auto& e : problem.entries())
        e.x().resize(max, 0);

    return problem;
}

Problem readProblem(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("File " + path + " doesn't exist!");
    }
    return readProblem(input);
}

double checkData(const Model &model, const Problem& p) {
    size_t errors = 0;
    for (size_t i = 0; i < p.entries().size(); i++) {
        double res = model.predict(p[i].x());
        if (res * p[i].y() < 0) { errors++; }
    }

    return static_cast<double>(errors) / static_cast<double>(p.entries().size());
}

Problem generateNormalData(size_t objects,
                           size_t features,
                           double stddiv,
                           double offset,
                           const std::string &seed) {
    Problem res;
    std::seed_seq s(seed.begin(), seed.end());
    std::default_random_engine re(s);
    std::normal_distribution<double> nd(0, stddiv);


    for (size_t oCount = 0; oCount < objects; oCount++) {
        Vector tempFeatures(features);
        for (size_t fCount = 0; fCount < features; fCount++) {
            tempFeatures[fCount] = nd(re) + (oCount % 2 == 0 ? offset : -offset);
        }
        res.add(Entry(oCount % 2 == 0 ? 1 : -1, std::move(tempFeatures)));
    }
    return res;
}

void logInfoToFile(std::vector<Info> stats, std::string path) {
    std::ofstream log;
    log.open(path);
    int c = 0;
    for (auto i : stats) {
        c++;
        log << "Exp #" << c << " {" << std::endl;
        log << "\tobjects:      " << i.objects << std::endl;
        log << "\tfeatures:     " << i.features << std::endl;
        log << "\tsteps:        " << i.steps << std::endl;
        log << "\tprecision:    " << i.precision << std::endl;
        log << "\tc:            " << i.c << std::endl;
        log << "\tw:            ";
        for (auto wi : i.w) log << wi << ", ";
        log << std::endl;
        log << "}" << std::endl << std::endl;
    }
    log.close();
}

} // namespace lc
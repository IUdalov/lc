#include "utils/utils.h"

#include "debug.h"

#include <sstream>
#include <regex>
#include <fstream>
#include <random>
#include <map>

namespace lc {

LossFunction lossFunctionByName(const std::string &name) {
    static std::map<std::string, LossFunction> data = {
            {loss_functions::V.name(),  loss_functions::V},
            {loss_functions::Q.name(),  loss_functions::Q},
            {loss_functions::Q3.name(), loss_functions::Q3},
    };
    return data.find(name)->second;
}

Problem readProblem(std::istream &content) {
    std::string line;

    Problem problem;
    std::regex tokenRegex("[0-9\\+\\-\\.:]+");
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
            x.resize(ind + 1, 0);
            x[ind] = val;
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

int readSVMFile(const std::string &path, lc::Objects &data, lc::Vector &classes) {
    (void) path;
    (void) data;
    (void) classes;
    throw std::runtime_error("Unimplemented");
}

int writeSVMFile(const std::string &path,
                 const lc::Objects &data,
                 const lc::Vector &classes) {
    if (data.size() != classes.size()) {
        return 1;
    }

    std::ofstream out;
    out.open(path);
    for (size_t i = 0; i < data.size(); i++) {
        if (classes[i] == 1) {
            out << "+1 ";
        } else {
            out << "-1 ";
        }
        for (size_t j = 0; j < data[i].size(); j++) {
            out << (j + 1) << ":" << data[i][j] << " ";
        }
        out << std::endl;
    }

    out.close();
    return 0;
}

double checkData(const Model &model, const Objects data, const Vector classes) {
    size_t errors = 0;
    for (size_t i = 0; i < data.size(); i++) {
        double res = model.predict(data[i]);
        if (res * classes[i] < 0) { errors++; }
    }

    return static_cast<double>(errors) / static_cast<double>(data.size());
}

double checkData(const Model &model, const Problem& p) {
    size_t errors = 0;
    for (size_t i = 0; i < p.entries().size(); i++) {
        double res = model.predict(p[i].x());
        if (res * p[i].y() < 0) { errors++; }
    }

    return static_cast<double>(errors) / static_cast<double>(p.entries().size());
}

void generateNormalData(Objects &o, Vector &c, size_t objects, size_t features, double stddiv, double offset,
                        const std::string &seed) {
    o.clear();
    c.clear();

    o.reserve(objects);
    c.reserve(objects);
    Vector tempFeatures(features);

    std::seed_seq s(seed.begin(), seed.end());
    std::default_random_engine re(s);
    std::normal_distribution<double> nd(0, stddiv);


    for (size_t oCount = 0; oCount < objects; oCount++) {
        for (size_t fCount = 0; fCount < features; fCount++) {
            tempFeatures[fCount] = nd(re) + (oCount % 2 == 0 ? offset : -offset);
        }
        o.push_back(tempFeatures);
        c.push_back(oCount % 2 == 0 ? 1 : -1);
    }
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
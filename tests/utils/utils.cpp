#include "utils/utils.h"

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <random>

using namespace lc;

std::string dataPath(DataSets dataset) {
    std::string prefix = "data/";
    switch(dataset) {
        case DataSets::iris:
            return prefix + "iris.csv";
        case DataSets::irisSimple:
            return prefix + "iris.simplified.csv";
        case DataSets::wine:
            return  prefix + "data.txt";
        case DataSets::easy:
            return  prefix + "easy.csv";
        case DataSets::easy2:
            return  prefix + "easy2.csv";
        case DataSets::easy3:
            return  prefix + "easy3.csv";
        case DataSets::easy4:
            return  prefix + "easy4.csv";
        default:
            return {};
    }
}

std::vector<std::string> csvDatasets() {
    return {
            dataPath(DataSets::irisSimple),
            dataPath(DataSets::easy),
            dataPath(DataSets::easy2),
            dataPath(DataSets::easy3)
    };
}

// val, val, val, class = {-1, 1}; 
int readCSVFile(const std::string& path, Objects& data, Vector& classes) {
    std::ifstream csvFile(path);
    std::string line;
    if (!csvFile) {
        throw std::runtime_error("File " + path + "not found!");
    }
    while(std::getline(csvFile, line)) {
        std::string temp = "";
        Vector node;
        for(size_t i = 0; i < line.size(); i++) {
            switch(line[i]) {
                case ',':
                    node.push_back(atof(temp.c_str()));
                    temp = "";
                    break;
                case ';':
                    data.push_back(node);
                    classes.push_back(atoi(temp.c_str()));
                    node.clear();
                    temp = "";
                    break;
                case '\n':
                case '\t':
                case ' ':
                    break;
                default:
                    temp.push_back(line[i]);
                    break;
            }
        }
    }
    return 0;
}

size_t guessTaskSpace(const std::string& path) {
    std::ifstream svmFile(path);
    std::string line;
    if (!svmFile) {
        throw std::runtime_error("File " + path + "not found!");
    }
    size_t number = 0;
    while(std::getline(svmFile, line)) {
        if (line == "\n") {
            continue;
        }
        line.erase(0, 3);
        std::string acc;
        for(size_t i = 0; i < line.size(); i++) {
            switch(line[i]) {
                case '\n':
                case ' ':
                    acc.clear();
                    break;
                case ':': {
                        size_t tmp = std::stoi(acc);
                        number = number < tmp ? tmp : number;
                        acc.clear();
                    }
                    break;
                default:
                    acc.push_back(line[i]);
                    break;
            }
        }
    }
    if (number == 0) {
        throw std::runtime_error("Format error at " + path);
    }
    return number;
}

int readSVMFile(const std::string& path, lc::Objects& data, lc::Vector& classes) {
    data.clear();
    classes.clear();

    size_t taskSize = guessTaskSpace(path);
    std::ifstream svmFile(path);
    std::string line;
    if (!svmFile) {
        throw std::runtime_error("File " + path + "not found!");
    }
    while(std::getline(svmFile, line)) {
        if (line == "\n") {
            continue;
        }
        int c = 0;
        switch(line[0]) {
            case '-': c = -1; break;
            case '+': c =  1; break;
            default: return 1;
        }
        line.erase(0, 3);
        Vector o(taskSize, NAN);
        std::string acc;
        size_t number;
        for(size_t i = 0; i < line.size(); i++) {
            switch(line[i]) {
                case '\n':
                case ' ':
                    o[number - 1] = std::stod(acc);
                    acc.clear();
                    break;
                case ':':
                    number = std::stoi(acc);
                    acc.clear();
                    break;
                default:
                    acc.push_back(line[i]);
                    break;
            }
        }
        bool isFull = true;
        for(auto i = 0; i < o.size(); i++)
            if (o[i] != o[i]) {
                isFull = false; break;
            }
        if (isFull) {
            data.push_back(o);
            classes.push_back(c);
        }
    }
    return 0;
}

int writeSVMFile(const std::string& path,
                 const lc::Objects& data,
                 const lc::Vector& classes) {
    if (data.size() != classes.size()) {
        return 1;
    }

    std::ofstream out;
    out.open(path);
    for(size_t i = 0; i < data.size(); i++) {
        if (classes[i] == 1) {
            out << "+1 ";
        } else {
            out << "-1 ";
        }
        for(size_t j = 0; j < data[i].size(); j++) {
            out << (j + 1) << ":" << data[i][j] << " ";
        }
        out << std::endl;
    }

    out.close();
    return  0;
}

void addDim(lc::Objects& data) {
    for(size_t i = 0; i < data.size(); i++) {
        data[i].push_back(1);
    }
}

double checkData(const Model& model, const Objects data, const Vector classes) {
    size_t errors = 0;
    for(auto i = 0; i < data.size(); i++) {
        int res = model.predict(data[i]);
        if (res != classes[i]) {errors++;}
    }

    return static_cast<double>(errors)/static_cast<double>(data.size());
}

bool about(double a, double b) {
    return fabs(a - b) < 0.00000000001;
}

void generateNormalData(Objects& o, Vector& c, size_t objects, size_t features, double stddiv, double offset, std::string seed) {
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

void logInfoToFile(std::vector<Info> stats, std::string path) {
    std::ofstream log;
    log.open(path);
    int c = 0;
    for(auto i : stats) {
        c++;
        log << "Exp #" << c <<" {" << std::endl;
        log << "\tdescr:        " << i.descr << std::endl;
        log << "\tobjects:      " << i.objects << std::endl;
        log << "\tfeatures:     " << i.features << std::endl;
        log << "\tsteps:        " << i.steps << std::endl;
        log << "\tprecision:    " << i.precision << std::endl;
        log << "\tc:            " << i.c << std::endl;
        log << "\terrorsBefore: " << i.errorsBefore << std::endl;
        log << "\terrorsAfter:  " << i.errorsAfter << std::endl;
        log << "\tw:            "; for(auto wi : i.w) log << wi << ", "; log << std::endl;
        log << "}" << std::endl << std::endl;
    }
    log.close();
}
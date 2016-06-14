#pragma once

#include <lc.h>

#include <vector>
#include <string>

int readCSVFile(const std::string& path,
                lc::Objects& data,
                lc::Vector& classes);

int readSVMFile(const std::string& path,
                lc::Objects& data,
                lc::Vector& classes);

int writeSVMFile(const std::string& path,
                 const lc::Objects& data,
                 const lc::Vector& classes);

void addDim(lc::Objects& data);

double checkData(
        const lc::Model& model,
        const lc::Objects data,
        const lc::Vector classes);

bool about(double a, double b);

// Generates normally distributed data set.
// With mean 0.
void generateNormalData(
        lc::Objects& o,
        lc::Vector& c,
        size_t objects,
        size_t features,
        double stddiv, // Standard deviation.
        double offset, // Offset for each feature. offset * class
        std::string seed = "random seed");

void logInfoToFile(std::vector<lc::Info> stats, std::string path);

// DEPRECATED: to refactor ----------------------------------------------------
enum class DataSets {
    iris,
    irisSimple,
    wine,
    easy,
    easy2,

// easy3:
// 10.times({ println "${r.nextGaussian() - 0.3}, ${r.nextGaussian() + 0.3}, 1;" })
// 10.times({ println "${r.nextGaussian() + 0.3}, ${r.nextGaussian() - 0.3}, -1;" })
            easy3,

// easy4
// 20.times({ println "${r.nextGaussian() + 1}, ${r.nextGaussian() + 1}, 1;" })
// 20.times({ println "${r.nextGaussian() - 1}, ${r.nextGaussian() - 1}, -1;" })
            easy4
};

std::string dataPath(DataSets dataset);

std::vector<std::string> csvDatasets();

#pragma once

#include <lc.h>

#include <istream>
#include <vector>
#include <string>

namespace lc {

LossFunction lossFunctionByName(const std::string &name);

Problem readProblem(std::istream &content);
Problem readProblem(const std::string& path);

int readSVMFile(const std::string &path,
                lc::Objects &data,
                lc::Vector &classes);

int writeSVMFile(const std::string &path,
                 const lc::Objects &data,
                 const lc::Vector &classes);

double checkData(
        const lc::Model &model,
        const lc::Objects data,
        const lc::Vector classes);

double checkData(const Model &model, const Problem& p);

// Generates normally distributed data set.
// With mean 0.
void generateNormalData(
        lc::Objects &o,
        lc::Vector &c,
        size_t objects,
        size_t features,
        double stddiv, // Standard deviation.
        double offset, // Offset for each feature. offset * class
        const std::string &seed = "random seed");

// Generates normally distributed data set.
// With mean 0.
Problem generateNormalData(size_t objects,
                           size_t features,
                           double stddiv, // Standard deviation.
                           double offset, // Offset for each feature. offset * class
                           const std::string &seed = "SEED");

void logInfoToFile(std::vector<lc::Info> stats, std::string path);

} // namespace lc
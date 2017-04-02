#pragma once

#include <lc.h>

#include <istream>
#include <vector>
#include <string>

namespace lc {

LossFunction lossFunctionByName(const std::string& name);
Problem readProblem(std::istream& content);
Problem readProblem(const std::string& path);

double checkData(const Model& model, const Problem& p);

// Generates normally distributed data set.
// With mean 0.
Problem generateNormalData(size_t objects,
                           size_t features,
                           double stddiv, // Standard deviation.
                           double offset, // Offset for each feature. offset * class
                           const std::string &seed = "SEED");

void logInfoToFile(std::vector<lc::Info> stats, std::string path);

} // namespace lc
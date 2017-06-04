#include "debug.h"

#include <cmath>

namespace lc {
namespace internal {

#ifndef NDEBUG

void validate(double v) {
    if (isnan(v))
        throw std::runtime_error("NaN");
}

void printVector(const std::string& name, const Vector& v) {
    std::cout << name << ":";
    for (const auto& e : v)
        std::cout << ' ' << e;
    std::cout << std::endl;
}

void validate(const Vector& v, size_t space) {
    if (v.size() != space)
        throw std::runtime_error("Size mismatch");

    for (const auto& e : v)
        validate(e);
}

void validate(const Problem& p) {
    if (p.entries().empty())
        throw std::runtime_error("Problem is empty");

    size_t space = p[0].size();
    for (const auto& e : p.entries())
        validate(e.x(), space);
}

#else // NDEBUG

void validate(double) {}
void printVector(const std::string&, const Vector&) {}
void validate(const Vector&, size_t) {}
void validate(const Problem&) {}

#endif //NDEBUG

} } // namespace lc::internal
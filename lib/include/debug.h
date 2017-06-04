#pragma once

#include "data.h"

#include <iostream>

#ifdef NDEBUG
#   define DEBUG if (false) std::cout
#else
#   define DEBUG std::cout << __FUNCTION__ << ":" << __LINE__ << " "
#endif

namespace lc {
namespace internal {

void printVector(const std::string& name, const Vector& v);

void validate(double v);
void validate(const Problem& p);
void validate(const Vector& v, size_t space);

}} // namespace lc::internal
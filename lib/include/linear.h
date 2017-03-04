#pragma once

#include <vector>

namespace lc {

typedef std::vector<double> Vector;
typedef std::vector<Vector> Objects;

// Malicious --------------------------------------------------------------
double dot(const Vector& lf, const Vector& rf);
double length(const Vector& data);
double distance(const Vector& v1, const Vector& v2);
bool compare(double a, double b);

} // namespace lc
#include "lc.h"

#include "debug.h"

#include <math.h>
#include <limits>

namespace lc {

std::string getVersion() {
    return "0.1.0";
}

double length(const Vector& data) {
    double acc = 0;
    for(size_t i = 0; i < data.size(); i++) {
        acc += pow(data[i], 2);
    }
    return sqrt(acc);
}

double distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v1.size()) {
        throw std::runtime_error(std::string("Error in") + __FUNCTION__);
    }
    double res = 0;
    for(size_t i = 0; i < v1.size(); i++) {
        res += pow(v1[i] - v2[i], 2);
    }
    return sqrt(res);
}

double dot(const Vector& lf, const Vector& rf) {
    double acc = 0;
    for(size_t i = 0; i < lf.size(); i++) {
        acc += lf[i] * rf[i];
    }
    return acc;
}

bool compare(double a, double b) {
    return fabs(a - b) < 3*std::numeric_limits<double>::epsilon();
}

}
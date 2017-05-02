#include "lc.h"

#include "debug.h"

#include <iostream>
#include <limits>

#include <cassert>
#include <cmath>

namespace lc {

std::ostream& operator<<(std::ostream& out, const Entry& e) {
    out << (e.y() == 1 ? "+1" : "-1");
    for (size_t i = 0; i < e.x().size(); i++)
        if (e[i] != 0)
            out << " " << (i + 1) << ":" << e[i];

    out << std::endl;
    return out;
}

std::ostream& operator<<(std::ostream& out, const Problem& p) {
    out << "Problem {" << std::endl;
    out << "\tobjects = " << p.entries().size() << std::endl;
    out << "\tfeatures = " << p[0].x().size() << std::endl;
    out << "}" << std::endl;
    return out;
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
    return fabs(a - b) < 3 * std::numeric_limits<double>::epsilon();
}

bool compareWith(double a, double b, double pres) {
    return fabs(a - b) < pres;
}

void norm(Vector& a) {
    double len = length(a);
    assert(!compare(len, 0));
    for (auto& e : a) {
        e = e / len;
    }
}

}
#include "lc.h"

#include "debug.h"

#include <limits>
#include <regex>

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

namespace {

std::pair<size_t,double> parsePair(const std::string& src) {
    size_t pos = src.find(":");
    if (pos == std::string::npos)
        throw std::runtime_error("Unexpected value: " + src);

    return std::make_pair<size_t, double >(
            std::stoul(src.substr(0, pos)),
            std::stod(src.substr(pos + 1)));
};

void addNew(Vector& x, std::string& buf) {
    auto p = parsePair(buf);
    if (p.first > x.size())
        x.resize(p.first, 0);

    x[p.first - 1] = p.second;
    buf.clear();
}

} // namespace

std::istream& operator>>(std::istream& in, Problem& problem) {
    if (!in)
        throw std::runtime_error("malformed ifstream");

    std::string line;
    std::regex tokenRegex("[e0-9\\+\\-\\.:]+");

    Vector x;
    std::string buf;

    while (std::getline(in, line)) {
        x.assign(x.size(), 0.0);
        for(size_t i = 3; i < line.size(); i++)
            if (line[i] == ' ')
                addNew(x, buf);
            else
                buf.push_back(line[i]);

        addNew(x, buf);

        int y = 0;
        auto yStr = line.substr(0,2);
        if (yStr == "+1") y = 1;
        else if (yStr == "-1") y = -1;
        else throw std::runtime_error("Unexpected token: " + yStr);

        problem.add(Entry(y, x));
    }

    size_t max = 0;
    for(const auto& e : problem.entries())
        max = std::max(e.size(), max);
    for(auto& e : problem.entries())
        e.x().resize(max, 0);

    return in;
}

namespace internal {

double length(const Vector& data) {
    double acc = 0;
    for(size_t i = 0; i < data.size(); i++) {
        acc += pow(data[i], 2);
    }
    return sqrt(acc);
}

double distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v1.size()) {
        throw std::runtime_error(std::string("Error in ") + __FUNCTION__);
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

} } // namspace lc::internal
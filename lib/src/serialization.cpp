#include <lc.h>

#include <sstream>

namespace lc {

std::string toString(const Vector& v) {
    std::stringstream ss;
    for(size_t i = 0; i < v.size(); i++) {
        ss << " " << v[i];
    }
    return ss.str();
}

Vector fromString(const std::string& str) {
    Vector res;
    std::stringstream ss(str);
    while (ss) {
        double buf = 0;
        ss >> buf;
        if (ss)
            res.push_back(buf);
    }

    return res;
}

std::ostream& operator<<(std::ostream& out, const Scaler& p) {
    out << toString(p.factor_) << std::endl;
    out << toString(p.offset_) << std::endl;
    return out;
}

std::istream& operator>>(std::istream& in, Scaler& r) {
    std::string factor;
    std::string offset;
    std::getline(in, factor);
    std::getline(in, offset);

    r.factor_ = fromString(factor);
    r.offset_ = fromString(offset);

    if (!r) throw std::runtime_error("Fail to load model: invalid resizer");
    return in;
}

std::ostream& operator<<(std::ostream& out, const Model& m) {
    out << m.k_.name() << std::endl;
    out << toString(m.w_) << std::endl;
    if (m.scaler_)
        out << *m.scaler_;
    return out;
}

std::istream& operator>>(std::istream& in, Model& m) {
    std::string k;
    std::getline(in, k);
    m.kernel(kernels::fromName(k));

    std::string w;
    std::getline(in, w);
    m.w_ = fromString(w);

    if (!in.eof()) {
        m.scaler_.reset(new Scaler());
        in >> *m.scaler_;
    }
    return in;
}

} // namespace lc
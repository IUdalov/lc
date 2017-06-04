#pragma once

#include "data.h"

namespace lc {
namespace internal {

class Scaler {
public:
    Scaler() = default;
    Scaler(const Problem& p);
    void apply(Vector& v) const;
    void apply(Problem& p) const;

    operator bool() { return offset_.size() == factor_.size() && !factor_.empty(); };
private:
    Vector factor_;
    Vector offset_;

    friend std::ostream& operator<<(std::ostream&, const Scaler&);
    friend std::istream& operator>>(std::istream&, Scaler&);
};

std::ostream& operator<<(std::ostream& out, const Scaler& p);
std::istream& operator>>(std::istream& in, Scaler& p);

} } // namespace lc::internal

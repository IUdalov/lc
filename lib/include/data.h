#pragma once

#include <vector>

namespace lc {

typedef std::vector<double> Vector;

class Entry {
public:
    Entry(int y, Vector x) : y_(y), x_(x) {}
    Entry(Entry&&) = default;

    int y() const { return y_; }
    Vector& x() { return x_; }
    const Vector& x() const { return x_; }
    size_t size() const { return x_.size(); }

    double& operator[](size_t ind) { return x_[ind]; }
    const double& operator[](size_t ind) const { return x_[ind]; }
    operator bool() const { return y_ > 0; }
    bool operator==(const Entry& other) const { return y_ == other.y_ && x_ == other.x_; }
    bool operator!=(const Entry& other) const { return !operator==(other); }

private:
    int y_;
    Vector x_;

private:
    Entry(const Entry&) = delete;
};

std::ostream& operator<<(std::ostream&, const Entry&);

class Problem {
public:
    Problem() = default;
    Problem(Problem&&) = default;

    Entry& operator[](size_t ind) { return entries_[ind]; }
    const Entry& operator[](size_t ind) const { return entries_[ind]; }

    std::vector<Entry>& entries() { return entries_; };
    const std::vector<Entry>& entries() const { return entries_; };

    void add(Entry e) { entries_.emplace_back(std::move(e)); }
    Problem dup() const {
        Problem dup;
        for(const auto& e : entries_)
            dup.add(Entry(e.y(), e.x()));
        return dup;
    }
private:
    std::vector<Entry> entries_;

private:
    Problem(const Problem&) = delete;
};

std::istream& operator>>(std::istream&, Problem&);
std::ostream& operator<<(std::ostream&, const Problem&);

double dot(const Vector& lf, const Vector& rf);
double length(const Vector& data);
double distance(const Vector& v1, const Vector& v2);
bool compare(double a, double b);
bool compareWith(double a, double b, double pres);
void norm(Vector& a);

} // namespace lc
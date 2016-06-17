#include <lc.h>

#include <math.h>

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
        for(auto i = 0; i < v1.size(); i++) {
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
    bool isSame(double a, double b) {
        return fabs(a - b) < 0.00000000001;
    }

    void scaleData(Objects& x, double scaleValue, Vector& factor, Vector& offset) {
        // let's scale space X -> X'
        // x' = M(x + v)
        // x = (M^-1)x' - v

        size_t l = x.size();
        size_t n = x[0].size();

        factor.clear();
        offset.clear();
        factor.resize(n);
        offset.resize(n);

        Vector min(n, std::numeric_limits<double>::max());
        Vector max(n, std::numeric_limits<double>::lowest());

        for(size_t i = 0; i < l; i++) {
            for(size_t j = 0; j < n; j ++) {
                if (x[i][j] < min[j]) min[j] = x[i][j];
                if (x[i][j] > max[j]) max[j] = x[i][j];
            }
        }

        for(size_t j = 0; j < n; j++) {
            factor[j] = scaleValue / (max[j] - min[j]);
            offset[j] = - (max[j] + min[j]) / 2;
        }
        for(size_t i = 0; i < l; i++) {
            for(size_t j = 0; j < n; j ++) {
                x[i][j] = (x[i][j] + offset[j]) * factor[j];
            }
        }
    }

    void unscaleVector(Vector& v, const Vector& factor, const Vector& offset) {
        if (v.empty()) {
            throw std::runtime_error("Empty vector");
        }
        if (v.size() != factor.size() || v.size() != offset.size()) {
            throw std::runtime_error("Vector doesn't fit to scale");
        }
        for(size_t i = 0; i < v.size(); i++) {
            v[i] = v[i] / factor[i] - offset[i];
        }
    }
}
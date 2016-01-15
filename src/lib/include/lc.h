#pragma once

#include <string>
#include <vector>
#include <functional>

namespace lc {
    std::string getVersion();

    typedef std::vector<float> Node;
    typedef std::vector<std::pair<Node, int>> Problem;

    typedef std::function<float(float)> LossFunction;

    class Model {
    public:
        Model(LossFunction&, float _c = 1);
        ~Model();
        void train(const Problem&);
        int predict(const Node&) const;
        void save() const;
        void load();
    private:
        Node classifier;
        LossFunction lf;
        float c;
    private:
    };
}

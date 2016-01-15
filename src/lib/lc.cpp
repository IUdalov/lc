#include "lc.h"

#include <limits>
#include <exception>

#include "debug.h"

namespace lc {
    // supportive functions
    float diff(LossFunction& lf, float point,  float pres = 0.00001);

    std::string getVersion() {
        return "0.1.0";
    }

    Model::Model(LossFunction& _lf, float _c) : classifier({}), lf(_lf), c(_c) {
        FUNCTION_LOG;
    }

    Model::~Model() {
        FUNCTION_LOG;
    }

    void Model::train(const Problem& _problem) {
        FUNCTION_LOG;
        LOG << "C: " << c << std::endl;
        LOG << "Nodes numer: " << _problem.size() << std::endl;
        if (_problem.size() == 0) {
            throw std::runtime_error("Problem doesn't have entries");
        }

        // TODO: stub
        classifier = {1,0};

        return;
    }

    int Model::predict(const Node& value) const {
        FUNCTION_LOG;
        if (classifier.size() == 0
            || classifier.size() != value.size()
            || c <= 0) {
            throw std::runtime_error("Model not configured");
        }
        float dot = 0;
        for(size_t i = 0; i < classifier.size(); i++) {
            dot += classifier[i] * value[i];
        }
        return dot >= 0 ? 1 : -1;
    }

    void Model::save() const {
    }

    void Model::load() {
    }

    float diff(const LossFunction& lf, float point, float pres) {
        return (lf(point + pres) - lf(point - pres)) / (2*pres);
    }
}

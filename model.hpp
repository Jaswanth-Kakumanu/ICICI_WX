// model.hpp - the machine learning part: a logistic regression, written from
// scratch, that predicts P(customer accepts | who they are, market, spread).
//
//     P(accept) = sigmoid( w . x )
//
// Every quote and the customer's answer is stored as a training sample. At the end
// of each trading day the weights are refitted on all data so far (Newton's method),
// exactly like a bank retraining a pricing model overnight.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "config.hpp"
#include "market.hpp"

namespace fx {

constexpr int NUM_FEATURES = 30;
using FeatureVec = std::array<double, NUM_FEATURES>;

// Builds the feature vector by calling push(group, item, value) once per feature.
// The same function is used to fill the numbers AND to list the feature names, so
// the two can never get out of sync.
//
// Only the spread features depend on the quoted spread, and they all have the form
// (relative spread) x (something), so the model's logit is linear in the spread.
template <class Push>
inline void emitFeatures(const QuoteContext& c, double spreadBps, Push&& push) {
    const SegmentSpec& seg = SEGMENTS[c.segment];
    const double sRel = spreadBps / c.refBps;                 // spread vs. the rate sheet
    const double size = std::log(c.notional / seg.medianNotional);  // trade size vs. segment norm
    const double vol = c.vol - 1.0;
    const double loyalty = c.loyalty - 0.5;
    const double side = static_cast<double>(c.side);

    for (int i = 0; i < NUM_SEGMENTS; ++i) push("segment", SEGMENTS[i].name, i == c.segment ? 1.0 : 0.0);
    for (int i = 0; i < NUM_PAIRS; ++i) push("pair", PAIRS[i].name, i == c.pair ? 1.0 : 0.0);
    for (int i = 0; i < NUM_SEGMENTS; ++i) push("size*segment", SEGMENTS[i].name, i == c.segment ? size : 0.0);
    for (int i = 0; i < NUM_SEGMENTS; ++i) push("vol*segment", SEGMENTS[i].name, i == c.segment ? vol : 0.0);
    for (int i = 0; i < NUM_SEGMENTS; ++i) push("loyalty*segment", SEGMENTS[i].name, i == c.segment ? loyalty : 0.0);
    push("side", "", side);

    // --- everything below depends on the quoted spread ---
    for (int i = 0; i < NUM_SEGMENTS; ++i) push("spread*segment", SEGMENTS[i].name, i == c.segment ? sRel : 0.0);
    for (int i = 0; i < NUM_PAIRS; ++i) push("spread*pair", PAIRS[i].name, i == c.pair ? sRel : 0.0);
    push("spread*size", "", sRel * size);
    push("spread*vol", "", sRel * vol);
    push("spread*side", "", sRel * side);
    push("spread*loyalty", "", sRel * loyalty);
}

inline void buildFeatures(const QuoteContext& c, double spreadBps, FeatureVec& x) {
    int k = 0;
    emitFeatures(c, spreadBps, [&](const char*, const char*, double v) {
        assert(k < NUM_FEATURES);
        x[k++] = v;
    });
    assert(k == NUM_FEATURES);
}

inline std::vector<std::string> featureNames() {
    QuoteContext c;
    c.notional = SEGMENTS[0].medianNotional;
    std::vector<std::string> names;
    emitFeatures(c, 1.0, [&](const char* group, const char* item, double) {
        std::string n = group;
        if (*item) n += std::string(":") + item;
        names.push_back(n);
    });
    assert(static_cast<int>(names.size()) == NUM_FEATURES);
    return names;
}

class AcceptanceModel {
public:
    explicit AcceptanceModel(double ridge) : ridge_(ridge) { w_.fill(0.0); }

    double logit(const FeatureVec& x) const {
        double z = 0.0;
        for (int k = 0; k < NUM_FEATURES; ++k) z += w_[k] * x[k];
        return z;
    }

    // Predicted acceptance probability, clipped away from 0 and 1 so log-loss stays finite.
    double predict(const FeatureVec& x) const {
        const double p = sigmoid(logit(x));
        return std::min(1.0 - 1e-6, std::max(1e-6, p));
    }

    // Remember one observed outcome (accepted = true/false). Does not change the weights.
    void addSample(const FeatureVec& x, bool accepted) { data_.push_back({x, accepted ? 1.0 : 0.0}); }

    std::size_t numSamples() const { return data_.size(); }

    // Refit the weights on ALL samples seen so far (maximum likelihood + a small ridge penalty)
    // using Newton's method, a.k.a. iteratively re-weighted least squares (IRLS):
    //
    //     w  <-  w - H^-1 g
    //     g = sum_i (p_i - y_i) x_i + ridge * w                  (gradient of the loss)
    //     H = sum_i p_i (1 - p_i) x_i x_i^T + ridge * I          (Hessian of the loss)
    //
    // Starting from yesterday's weights it normally converges in 1-3 iterations.
    // Returns the number of iterations used.
    int refit(int maxIter = 12, double tol = 1e-5) {
        if (data_.empty()) return 0;
        constexpr int D = NUM_FEATURES;
        int iter = 0;
        for (; iter < maxIter; ++iter) {
            std::array<double, D> grad{};
            std::array<std::array<double, D>, D> hess{};
            for (const Sample& s : data_) {
                const double p = sigmoid(logit(s.x));
                const double err = p - s.y;
                const double wgt = p * (1.0 - p);
                for (int i = 0; i < D; ++i) {
                    if (s.x[i] == 0.0) continue;  // one-hot features are mostly zero
                    grad[i] += err * s.x[i];
                    const double wi = wgt * s.x[i];
                    for (int j = 0; j <= i; ++j) hess[i][j] += wi * s.x[j];
                }
            }
            for (int i = 0; i < D; ++i) {
                grad[i] += ridge_ * w_[i];
                hess[i][i] += ridge_;
                for (int j = i + 1; j < D; ++j) hess[i][j] = hess[j][i];  // mirror lower -> upper
            }
            const std::array<double, D> step = solveSpd(hess, grad);
            double maxStep = 0.0;
            for (int i = 0; i < D; ++i) {
                w_[i] -= step[i];
                maxStep = std::max(maxStep, std::fabs(step[i]));
            }
            if (maxStep < tol) {
                ++iter;
                break;
            }
        }
        return iter;
    }

    const std::array<double, NUM_FEATURES>& weights() const { return w_; }

private:
    struct Sample {
        FeatureVec x;
        double y;
    };

    // Solve A z = b for a symmetric positive-definite A using Cholesky decomposition.
    static std::array<double, NUM_FEATURES> solveSpd(const std::array<std::array<double, NUM_FEATURES>, NUM_FEATURES>& A,
                                                     const std::array<double, NUM_FEATURES>& b) {
        constexpr int D = NUM_FEATURES;
        std::array<std::array<double, D>, D> L{};
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = A[i][j];
                for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
                L[i][j] = (i == j) ? std::sqrt(std::max(sum, 1e-12)) : sum / L[j][j];
            }
        }
        std::array<double, D> y{}, z{};
        for (int i = 0; i < D; ++i) {  // forward substitution: L y = b
            double sum = b[i];
            for (int k = 0; k < i; ++k) sum -= L[i][k] * y[k];
            y[i] = sum / L[i][i];
        }
        for (int i = D - 1; i >= 0; --i) {  // back substitution: L^T z = y
            double sum = y[i];
            for (int k = i + 1; k < D; ++k) sum -= L[k][i] * z[k];
            z[i] = sum / L[i][i];
        }
        return z;
    }

    double ridge_;
    std::array<double, NUM_FEATURES> w_;
    std::vector<Sample> data_;
};

}  // namespace fx

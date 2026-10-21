// rng.hpp - small deterministic random number helper.
//
// std::normal_distribution and friends are allowed to produce different values
// on different compilers/standard libraries. We only rely on std::mt19937_64
// (whose output IS fully specified by the standard) and build the rest by hand,
// so the same seed gives the same simulation on every machine.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace fx {

constexpr double kPi = 3.14159265358979323846;

class Rng {
public:
    explicit Rng(std::uint64_t seed) : eng_(seed) {}

    // Uniform in [0, 1)
    double uniform() { return static_cast<double>(eng_() >> 11) * (1.0 / 9007199254740992.0); }

    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

    // Standard normal via Box-Muller (caches the second value).
    double normal() {
        if (hasSpare_) {
            hasSpare_ = false;
            return spare_;
        }
        double u1 = uniform();
        while (u1 <= 1e-300) u1 = uniform();
        const double u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        spare_ = r * std::sin(2.0 * kPi * u2);
        hasSpare_ = true;
        return r * std::cos(2.0 * kPi * u2);
    }

    // Draw an index in [0, n) with probability proportional to weights[i].
    int categorical(const double* weights, int n) {
        double total = 0.0;
        for (int i = 0; i < n; ++i) total += weights[i];
        double u = uniform() * total;
        for (int i = 0; i < n; ++i) {
            u -= weights[i];
            if (u < 0.0) return i;
        }
        return n - 1;
    }

private:
    std::mt19937_64 eng_;
    bool hasSpare_ = false;
    double spare_ = 0.0;
};

}  // namespace fx

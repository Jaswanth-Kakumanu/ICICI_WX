// market.hpp - the simulated world: customers, market conditions and the hidden
// "true" price-response behaviour that the pricing engine has to discover.
//
// KEY DESIGN IDEA: the whole day-by-day stream of customers is generated ONCE per
// seed, including a pre-drawn random number `u` per customer. Every strategy is then
// replayed on the identical stream (common random numbers), so differences in PnL
// come from pricing decisions and not from luck.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "config.hpp"
#include "rng.hpp"

namespace fx {

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// One quote request from one customer.
struct Trade {
    int day = 0;
    int segment = 0;
    int pair = 0;
    int side = +1;          // +1: customer BUYS foreign currency (bank goes short), -1: customer sells
    double notional = 0.0;  // USD
    double loyalty = 0.5;   // 0..1 relationship score - observable to the bank
    double vol = 1.0;       // market volatility regime (1 = normal) - observable
    double latent = 0.0;    // customer's hidden price-sensitivity shock - NOT observable
    double u = 0.0;         // pre-drawn uniform: customer accepts iff u < P(accept)
};

// What the pricing engine is allowed to see when it has to quote.
struct QuoteContext {
    int segment = 0;
    int pair = 0;
    int side = +1;
    double notional = 0.0;
    double loyalty = 0.5;
    double vol = 1.0;
    double refBps = 1.0;  // hand-set rate-sheet spread for this segment/pair
};

struct MarketStream {
    int days = 0;
    int tradesPerDay = 0;
    std::vector<Trade> trades;                            // days * tradesPerDay, day-major
    std::vector<double> dayVol;                           // volatility regime per day
    std::vector<std::array<double, NUM_PAIRS>> dayReturns;  // pair price move per day
};

inline double refBps(int segment, int pair) {
    return SEGMENTS[segment].sheetBps * PAIRS[pair].toleranceFactor;
}

inline QuoteContext contextOf(const Trade& t) {
    QuoteContext c;
    c.segment = t.segment;
    c.pair = t.pair;
    c.side = t.side;
    c.notional = t.notional;
    c.loyalty = t.loyalty;
    c.vol = t.vol;
    c.refBps = refBps(t.segment, t.pair);
    return c;
}

// ---------------------------------------------------------------------------
// Ground truth: how a customer reacts to a spread.
// ---------------------------------------------------------------------------

// Highest spread (bps) this customer is willing to pay.
inline double reservationBps(const Trade& t, double latent) {
    const SegmentSpec& s = SEGMENTS[t.segment];
    const PairSpec& p = PAIRS[t.pair];
    const double sizeEffect = std::pow(t.notional / s.medianNotional, -s.sizeElasticity);
    const double volEffect = 1.0 + 0.25 * (t.vol - 1.0);   // nervous markets: less shopping around
    const double sideEffect = (t.side > 0) ? 1.0 : 0.9;    // customers selling FX are a bit pickier
    const double loyaltyEffect = 0.75 + 0.5 * t.loyalty;   // loyal customers tolerate more
    return s.reservationBps * p.toleranceFactor * sizeEffect * volEffect * sideEffect *
           loyaltyEffect * std::exp(s.heterogeneity * latent);
}

inline double acceptWidth(const Trade& t) {
    return SEGMENTS[t.segment].widthBps * PAIRS[t.pair].toleranceFactor;
}

// Probability that this exact customer (including hidden shock) accepts `spreadBps`.
inline double acceptProb(const Trade& t, double spreadBps, double latent) {
    return sigmoid((reservationBps(t, latent) - spreadBps) / acceptWidth(t));
}

// Probability of acceptance averaged over the hidden shock, i.e. the best anyone
// can predict from observable information. 9-point Gauss-Hermite quadrature.
inline double marginalAcceptProb(const Trade& t, double spreadBps) {
    static const double X[5] = {0.0, 0.7235510187528376, 1.4685532892166679, 2.2665805845318431,
                                3.1909932017815276};
    static const double W[5] = {0.7202352156060510, 0.4326515590025558, 0.08847452739437657,
                                0.004943624275536947, 0.00003960697726326};
    const double k = 1.0 / std::sqrt(kPi);
    const double root2 = std::sqrt(2.0);
    double p = W[0] * k * acceptProb(t, spreadBps, 0.0);
    for (int i = 1; i < 5; ++i) {
        p += W[i] * k * acceptProb(t, spreadBps, +root2 * X[i]);
        p += W[i] * k * acceptProb(t, spreadBps, -root2 * X[i]);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Stream generation
// ---------------------------------------------------------------------------

inline MarketStream generateStream(std::uint64_t seed, int days, int tradesPerDay,
                                   const SimConfig& cfg) {
    Rng rng(seed * 0x9E3779B97F4A7C15ULL + 12345ULL);

    double segShare[NUM_SEGMENTS], pairShare[NUM_PAIRS];
    for (int i = 0; i < NUM_SEGMENTS; ++i) segShare[i] = SEGMENTS[i].share;
    for (int i = 0; i < NUM_PAIRS; ++i) pairShare[i] = PAIRS[i].share;

    MarketStream m;
    m.days = days;
    m.tradesPerDay = tradesPerDay;
    m.trades.reserve(static_cast<std::size_t>(days) * tradesPerDay);
    m.dayVol.resize(days);
    m.dayReturns.resize(days);

    double logVol = 0.0;
    for (int d = 0; d < days; ++d) {
        // Volatility regime: mean-reverting log process, clamped to a sane range.
        logVol = 0.85 * logVol + 0.20 * rng.normal();
        const double vol = std::min(2.5, std::max(0.6, std::exp(logVol)));
        m.dayVol[d] = vol;

        // Daily price move and how one-sided customer flow is today, per pair.
        double buyProb[NUM_PAIRS];
        for (int p = 0; p < NUM_PAIRS; ++p) {
            m.dayReturns[d][p] = PAIRS[p].dailyVol * vol * rng.normal();
            const double bias = std::min(0.35, std::max(-0.35, cfg.flowBiasSigma * rng.normal()));
            buyProb[p] = 0.5 + bias;
        }

        for (int i = 0; i < tradesPerDay; ++i) {
            Trade t;
            t.day = d;
            t.segment = rng.categorical(segShare, NUM_SEGMENTS);
            t.pair = rng.categorical(pairShare, NUM_PAIRS);
            t.side = (rng.uniform() < buyProb[t.pair]) ? +1 : -1;
            const SegmentSpec& s = SEGMENTS[t.segment];
            t.notional = s.medianNotional * std::exp(s.sigmaNotional * rng.normal());
            t.notional = std::min(5'000'000.0, std::max(200.0, t.notional));
            t.loyalty = rng.uniform();
            t.vol = vol;
            t.latent = rng.normal();
            t.u = rng.uniform();
            m.trades.push_back(t);
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Static rate sheets (the baseline the dynamic engine has to beat)
// ---------------------------------------------------------------------------

using RateSheet = std::array<std::array<double, NUM_PAIRS>, NUM_SEGMENTS>;  // spread in bps

// The hand-set sheet: one spread per segment, scaled per pair.
inline RateSheet handSetRateSheet() {
    RateSheet sheet{};
    for (int s = 0; s < NUM_SEGMENTS; ++s)
        for (int p = 0; p < NUM_PAIRS; ++p) sheet[s][p] = refBps(s, p);
    return sheet;
}

// A much stronger baseline: for every (segment, pair) cell, pick the fixed spread that
// would have earned the most on a historical calibration stream (think of it as the
// pricing desk having run A/B tests). Calibration uses a DIFFERENT seed than the
// evaluation, so the tuned sheet is not cheating.
inline RateSheet tuneRateSheet(const MarketStream& calib, double lo = 0.30, double hi = 2.00,
                               double step = 0.01) {
    const int K = static_cast<int>(std::round((hi - lo) / step)) + 1;
    std::vector<double> score(static_cast<std::size_t>(NUM_SEGMENTS) * NUM_PAIRS * K, 0.0);

    for (const Trade& t : calib.trades) {
        const double ref = refBps(t.segment, t.pair);
        const double r = reservationBps(t, t.latent);
        const double w = acceptWidth(t);
        double* cell = &score[(static_cast<std::size_t>(t.segment) * NUM_PAIRS + t.pair) * K];
        for (int k = 0; k < K; ++k) {
            const double spread = ref * (lo + k * step);
            if (t.u < sigmoid((r - spread) / w)) cell[k] += spread / 1e4 * t.notional;
        }
    }

    RateSheet sheet{};
    for (int s = 0; s < NUM_SEGMENTS; ++s) {
        for (int p = 0; p < NUM_PAIRS; ++p) {
            const double* cell = &score[(static_cast<std::size_t>(s) * NUM_PAIRS + p) * K];
            int best = 0;
            for (int k = 1; k < K; ++k)
                if (cell[k] > cell[best]) best = k;
            sheet[s][p] = refBps(s, p) * (lo + best * step);
        }
    }
    return sheet;
}

}  // namespace fx

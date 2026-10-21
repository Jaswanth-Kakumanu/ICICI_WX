// strategy.hpp - the pricing brains.
//
//   StaticStrategy   : looks the spread up in a rate sheet (what banks do today).
//   DynamicStrategy  : learns customer price-acceptance online and picks the spread
//                      that maximises expected profit for THIS customer and THIS trade,
//                      optionally penalising trades that push inventory the wrong way.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

#include "config.hpp"
#include "market.hpp"
#include "model.hpp"
#include "rng.hpp"

namespace fx {

using Positions = std::array<double, NUM_PAIRS>;  // bank's open position per pair, USD

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual std::string name() const = 0;

    // Decide the spread (bps) to quote.
    virtual double quote(const QuoteContext& c, const Positions& pos, int day) = 0;

    // Learn from the customer's answer.
    virtual void observe(const QuoteContext&, double /*spreadBps*/, bool /*accepted*/, int /*day*/) {}

    // Called after the last quote of every day (e.g. to retrain overnight).
    virtual void endOfDay(int /*day*/) {}

    // Model's predicted acceptance probability, or NaN if the strategy has no model.
    virtual double predictAccept(const QuoteContext&, double /*spreadBps*/) const {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

// ---------------------------------------------------------------------------

class StaticStrategy : public Strategy {
public:
    StaticStrategy(std::string label, const RateSheet& sheet) : label_(std::move(label)), sheet_(sheet) {}

    std::string name() const override { return label_; }

    double quote(const QuoteContext& c, const Positions&, int) override {
        return sheet_[c.segment][c.pair];
    }

private:
    std::string label_;
    RateSheet sheet_;
};

// ---------------------------------------------------------------------------

class DynamicStrategy : public Strategy {
public:
    DynamicStrategy(std::string label, const DynamicParams& params, int warmupDays, std::uint64_t seed)
        : label_(std::move(label)),
          p_(params),
          warmupDays_(warmupDays),
          model_(params.ridge),
          rng_(seed * 0xD1B54A32D192ED03ULL + 777ULL) {}

    std::string name() const override { return label_; }

    double quote(const QuoteContext& c, const Positions& pos, int day) override {
        // 1) Exploration. A model can only learn a price-response curve if it has seen
        //    customers react to many different prices, so we quote randomly during the
        //    warm-up days and keep exploring a small share of quotes afterwards.
        if (day < warmupDays_) return c.refBps * rng_.uniform(p_.warmupLo, p_.warmupHi);
        if (rng_.uniform() < p_.epsilon) return c.refBps * rng_.uniform(p_.exploreLo, p_.exploreHi);

        // 2) Exploitation. Try every candidate spread and keep the one with the best
        //    expected profit:  P(accept | spread) * (revenue - inventory cost).
        const double inventoryCost = p_.inventoryAware ? inventoryPenaltyUsd(c, pos) : 0.0;
        FeatureVec x;
        double bestSpread = c.refBps;
        double bestValue = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < p_.gridSteps; ++k) {
            const double spread = c.refBps * (p_.minMult + k * (p_.maxMult - p_.minMult) / (p_.gridSteps - 1));
            buildFeatures(c, spread, x);
            const double revenue = spread / 1e4 * c.notional;
            const double value = model_.predict(x) * (revenue - inventoryCost);
            if (value > bestValue) {
                bestValue = value;
                bestSpread = spread;
            }
        }
        return bestSpread;
    }

    void observe(const QuoteContext& c, double spreadBps, bool accepted, int) override {
        FeatureVec x;
        buildFeatures(c, spreadBps, x);
        model_.addSample(x, accepted);
    }

    // Overnight retraining on everything seen so far.
    void endOfDay(int) override { model_.refit(); }

    double predictAccept(const QuoteContext& c, double spreadBps) const override {
        FeatureVec x;
        buildFeatures(c, spreadBps, x);
        return model_.predict(x);
    }

    const AcceptanceModel& model() const { return model_; }

private:
    // Extra cost (USD) of taking this trade given the current open position in the pair.
    // Positive if the trade makes the position bigger (we will probably pay to hedge it),
    // negative if it shrinks the position (we are happy to win that flow).
    //
    //   hedging term : theta * hedgeCost * (|position after| - |position before|)
    //   risk term    : 0.5 * gamma * sigma^2 * (position after^2 - position before^2)
    double inventoryPenaltyUsd(const QuoteContext& c, const Positions& pos) const {
        const PairSpec& pair = PAIRS[c.pair];
        const double before = pos[c.pair];
        const double after = before - c.side * c.notional;  // customer buys -> bank goes short
        const double hedgeRate = pair.hedgeBps * c.vol / 1e4;
        const double sigma = pair.dailyVol * c.vol;
        const double hedgingTerm = p_.nettingTheta * hedgeRate * (std::fabs(after) - std::fabs(before));
        const double riskTerm = 0.5 * p_.riskAversion * sigma * sigma * (after * after - before * before);
        return hedgingTerm + riskTerm;
    }

    std::string label_;
    DynamicParams p_;
    int warmupDays_;
    AcceptanceModel model_;
    Rng rng_;
};

}  // namespace fx

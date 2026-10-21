// engine.hpp - replays one market stream against one strategy and keeps the books.
//
// Accounting (all in USD):
//   spread revenue : spread(bps)/10000 * notional, for every accepted quote
//   hedge cost     : cost of unwinding inventory in the interbank market
//                      - forced hedge if a pair's position breaches the hard limit
//                      - all positions are flattened at the end of every day
//   MTM PnL        : position * daily price move (risk that averages to zero)
//
// The headline "PnL" of the project is  spread revenue - hedge cost  (called corePnl).
// MTM is tracked separately: it is pure market noise, but its spread across days shows
// how much price risk each strategy carried.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "config.hpp"
#include "market.hpp"
#include "strategy.hpp"

namespace fx {

struct RunResult {
    std::string name;

    double spreadRevenue = 0.0;
    double hedgeCost = 0.0;
    double mtmPnl = 0.0;
    double warmupPnl = 0.0;  // corePnl earned during warm-up days
    double steadyPnl = 0.0;  // corePnl earned after warm-up

    long quotes = 0;
    long accepts = 0;
    double acceptedNotional = 0.0;
    double hedgedNotional = 0.0;
    double avgEodInventory = 0.0;  // mean over days of sum_pairs |position| before flattening
    double mtmStd = 0.0;           // std-dev of daily MTM PnL

    // Per-segment statistics over the steady-state days only.
    std::array<long, NUM_SEGMENTS> segQuotes{};
    std::array<long, NUM_SEGMENTS> segAccepts{};
    std::array<double, NUM_SEGMENTS> segSpreadSum{};

    // Daily series.
    std::vector<double> dailyPnl;        // corePnl per day
    std::vector<double> dailyInventory;  // sum_pairs |position| at end of day (before flatten)
    std::vector<double> dailyModelLoss;  // mean log-loss of the strategy's own predictions (NaN if none)
    std::vector<double> dailyOracleLoss; // mean log-loss of the best possible predictor

    double corePnl() const { return spreadRevenue - hedgeCost; }
    double totalPnl() const { return corePnl() + mtmPnl; }
    double acceptRate() const { return quotes ? static_cast<double>(accepts) / quotes : 0.0; }
};

inline double logLoss(double p, bool y) {
    p = std::min(1.0 - 1e-6, std::max(1e-6, p));
    return -(y ? std::log(p) : std::log(1.0 - p));
}

inline RunResult simulate(const MarketStream& market, Strategy& strategy, const SimConfig& cfg) {
    RunResult r;
    r.name = strategy.name();
    r.dailyPnl.assign(market.days, 0.0);
    r.dailyInventory.assign(market.days, 0.0);
    r.dailyModelLoss.assign(market.days, std::nan(""));
    r.dailyOracleLoss.assign(market.days, std::nan(""));

    Positions pos{};  // open position per pair, USD (positive = long the foreign currency)
    std::vector<double> dailyMtm(market.days, 0.0);

    for (int d = 0; d < market.days; ++d) {
        const bool steady = d >= cfg.warmupDays;
        double dayRevenue = 0.0, dayHedge = 0.0;
        double modelLoss = 0.0, oracleLoss = 0.0;
        int modelCount = 0;

        for (int i = 0; i < market.tradesPerDay; ++i) {
            const Trade& t = market.trades[static_cast<std::size_t>(d) * market.tradesPerDay + i];
            const QuoteContext ctx = contextOf(t);

            // 1. The strategy quotes a spread.
            const double spread = strategy.quote(ctx, pos, d);

            // 2. The customer answers (decided by the hidden true behaviour).
            const bool accepted = t.u < acceptProb(t, spread, t.latent);

            // 3. Diagnostics: how good were the strategy's predictions? (before it learns)
            const double predicted = strategy.predictAccept(ctx, spread);
            if (!std::isnan(predicted)) {
                modelLoss += logLoss(predicted, accepted);
                oracleLoss += logLoss(marginalAcceptProb(t, spread), accepted);
                ++modelCount;
            }

            // 4. The strategy learns from the outcome.
            strategy.observe(ctx, spread, accepted, d);

            // 5. Bookkeeping.
            ++r.quotes;
            if (steady) {
                ++r.segQuotes[t.segment];
                r.segSpreadSum[t.segment] += spread;
            }
            if (!accepted) continue;

            ++r.accepts;
            r.acceptedNotional += t.notional;
            if (steady) ++r.segAccepts[t.segment];
            dayRevenue += spread / 1e4 * t.notional;

            // Customer buys foreign currency -> bank is short it (and vice versa).
            pos[t.pair] -= t.side * t.notional;

            // Hard limit: hedge back to a safe level and pay the interbank cost.
            if (std::fabs(pos[t.pair]) > cfg.hardLimitUsd) {
                const double target = cfg.hardLimitUsd * cfg.hedgeBackFraction;
                const double amount = std::fabs(pos[t.pair]) - target;
                dayHedge += PAIRS[t.pair].hedgeBps * t.vol / 1e4 * amount;
                r.hedgedNotional += amount;
                pos[t.pair] = std::copysign(target, pos[t.pair]);
            }
        }

        strategy.endOfDay(d);

        // End of day: record inventory, take the price move, flatten everything.
        double inventory = 0.0, mtm = 0.0;
        for (int p = 0; p < NUM_PAIRS; ++p) {
            inventory += std::fabs(pos[p]);
            mtm += pos[p] * market.dayReturns[d][p];
            dayHedge += PAIRS[p].hedgeBps * market.dayVol[d] / 1e4 * std::fabs(pos[p]);
            r.hedgedNotional += std::fabs(pos[p]);
            pos[p] = 0.0;
        }

        r.spreadRevenue += dayRevenue;
        r.hedgeCost += dayHedge;
        r.mtmPnl += mtm;
        dailyMtm[d] = mtm;
        r.dailyPnl[d] = dayRevenue - dayHedge;
        r.dailyInventory[d] = inventory;
        (steady ? r.steadyPnl : r.warmupPnl) += r.dailyPnl[d];
        if (modelCount > 0) {
            r.dailyModelLoss[d] = modelLoss / modelCount;
            r.dailyOracleLoss[d] = oracleLoss / modelCount;
        }
    }

    double invSum = 0.0, mtmSum = 0.0;
    for (int d = 0; d < market.days; ++d) {
        invSum += r.dailyInventory[d];
        mtmSum += dailyMtm[d];
    }
    r.avgEodInventory = invSum / market.days;
    const double mtmMean = mtmSum / market.days;
    double var = 0.0;
    for (int d = 0; d < market.days; ++d) var += (dailyMtm[d] - mtmMean) * (dailyMtm[d] - mtmMean);
    r.mtmStd = std::sqrt(var / std::max(1, market.days - 1));
    return r;
}

}  // namespace fx

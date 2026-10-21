// main.cpp - runs the full experiment.
//
// For each random seed:
//   1. generate a market stream (customers, volatility, price moves)
//   2. generate a separate calibration stream and tune the static rate sheet on it
//   3. replay the SAME market stream against four strategies
//        static (hand-set) | static (tuned) | dynamic | dynamic + inventory-aware
// Then report mean / std of PnL and the uplift versus the tuned static sheet, and
// export everything for the web dashboard.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "engine.hpp"
#include "json.hpp"
#include "market.hpp"
#include "model.hpp"
#include "strategy.hpp"

using namespace fx;

namespace {

struct Options {
    int seeds = 10;
    int firstSeed = 1;
    int days = 60;
    int trades = 500;
    std::string outDir = "dashboard";
    double riskAversion = -1.0;  // <0: keep the default from config.hpp
    double nettingTheta = -1.0;
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--seeds") o.seeds = std::atoi(next());
        else if (a == "--first-seed") o.firstSeed = std::atoi(next());
        else if (a == "--days") o.days = std::atoi(next());
        else if (a == "--trades") o.trades = std::atoi(next());
        else if (a == "--out") o.outDir = next();
        else if (a == "--gamma") o.riskAversion = std::atof(next());
        else if (a == "--theta") o.nettingTheta = std::atof(next());
        else {
            std::fprintf(stderr,
                         "usage: fxprice [--seeds N] [--first-seed S] [--days D] [--trades T]\n"
                         "               [--out DIR] [--gamma G] [--theta T]\n");
            std::exit(a == "--help" ? 0 : 1);
        }
    }
    return o;
}

struct StrategyInfo {
    const char* id;
    const char* label;
};
constexpr int NUM_STRATEGIES = 4;
const StrategyInfo STRATEGIES[NUM_STRATEGIES] = {
    {"static_sheet", "Static rate sheet (hand-set)"},
    {"static_tuned", "Static rate sheet (tuned)"},
    {"dynamic", "Dynamic pricing"},
    {"dynamic_inv", "Dynamic + inventory-aware"},
};
constexpr int TUNED = 1;  // the baseline that uplifts are measured against

struct AcceptanceCurve {
    std::string segment;
    double refBps = 0.0;
    std::vector<double> mult, learned, truth;
};

struct SeedOutcome {
    std::uint64_t seed = 0;
    std::array<RunResult, NUM_STRATEGIES> runs;
};

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

double minOf(const std::vector<double>& v) {
    double m = v.empty() ? 0.0 : v[0];
    for (double x : v) m = std::min(m, x);
    return m;
}

double maxOf(const std::vector<double>& v) {
    double m = v.empty() ? 0.0 : v[0];
    for (double x : v) m = std::max(m, x);
    return m;
}

double pct(double value, double baseline) { return baseline != 0.0 ? 100.0 * (value - baseline) / baseline : 0.0; }

// Mean across seeds of a per-day series (NaN days are skipped).
std::vector<double> meanSeries(const std::vector<SeedOutcome>& out, int strat,
                               const std::vector<double> RunResult::*member) {
    const std::size_t n = (out[0].runs[strat].*member).size();
    std::vector<double> m(n, std::nan(""));
    for (std::size_t d = 0; d < n; ++d) {
        double s = 0.0;
        int c = 0;
        for (const auto& so : out) {
            const double x = (so.runs[strat].*member)[d];
            if (!std::isnan(x)) {
                s += x;
                ++c;
            }
        }
        if (c) m[d] = s / c;
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);

    SimConfig cfg;
    cfg.days = opt.days;
    cfg.tradesPerDay = opt.trades;
    cfg.warmupDays = std::min(cfg.warmupDays, std::max(0, opt.days / 3));

    DynamicParams plain;  // dynamic pricing without inventory awareness
    DynamicParams withInv;
    withInv.inventoryAware = true;
    if (opt.riskAversion >= 0.0) withInv.riskAversion = opt.riskAversion;
    if (opt.nettingTheta >= 0.0) withInv.nettingTheta = opt.nettingTheta;

    std::printf("FX pricing engine | %d seeds | %d days x %d quotes | %d warm-up days\n\n", opt.seeds,
                cfg.days, cfg.tradesPerDay, cfg.warmupDays);

    std::vector<SeedOutcome> outcomes;
    std::vector<AcceptanceCurve> curves;
    std::vector<std::string> weightNames = featureNames();
    std::vector<double> finalWeights;

    for (int i = 0; i < opt.seeds; ++i) {
        const std::uint64_t seed = static_cast<std::uint64_t>(opt.firstSeed + i);

        // Evaluation stream, plus a longer independent history for tuning the static sheet.
        const MarketStream market = generateStream(seed, cfg.days, cfg.tradesPerDay, cfg);
        const MarketStream history = generateStream(9000 + seed, cfg.days * 2, cfg.tradesPerDay, cfg);
        const RateSheet tuned = tuneRateSheet(history);

        StaticStrategy sHand("static_sheet", handSetRateSheet());
        StaticStrategy sTuned("static_tuned", tuned);
        DynamicStrategy dPlain("dynamic", plain, cfg.warmupDays, seed);
        DynamicStrategy dInv("dynamic_inv", withInv, cfg.warmupDays, seed);

        SeedOutcome so;
        so.seed = seed;
        so.runs[0] = simulate(market, sHand, cfg);
        so.runs[1] = simulate(market, sTuned, cfg);
        so.runs[2] = simulate(market, dPlain, cfg);
        so.runs[3] = simulate(market, dInv, cfg);
        outcomes.push_back(so);

        // For the first seed, keep the learned price-response curves to compare with the truth.
        if (i == 0) {
            finalWeights.assign(dInv.model().weights().begin(), dInv.model().weights().end());
            for (int s = 0; s < NUM_SEGMENTS; ++s) {
                Trade t;  // a typical customer: median size, USD/INR, normal vol, average loyalty
                t.segment = s;
                t.pair = 0;
                t.side = +1;
                t.notional = SEGMENTS[s].medianNotional;
                t.loyalty = 0.5;
                t.vol = 1.0;
                const QuoteContext ctx = contextOf(t);
                AcceptanceCurve c;
                c.segment = SEGMENTS[s].name;
                c.refBps = ctx.refBps;
                for (double m = 0.30; m <= 2.0001; m += 0.05) {
                    c.mult.push_back(m);
                    c.learned.push_back(dInv.predictAccept(ctx, ctx.refBps * m));
                    c.truth.push_back(marginalAcceptProb(t, ctx.refBps * m));
                }
                curves.push_back(c);
            }
        }
    }

    // ---- aggregate ----
    std::array<std::vector<double>, NUM_STRATEGIES> core, steady, total, mtmStd, inv, hedge, accept;
    std::array<std::vector<double>, NUM_STRATEGIES> upliftOverall, upliftSteady;  // vs tuned static
    for (const SeedOutcome& so : outcomes) {
        for (int s = 0; s < NUM_STRATEGIES; ++s) {
            const RunResult& r = so.runs[s];
            core[s].push_back(r.corePnl());
            steady[s].push_back(r.steadyPnl);
            total[s].push_back(r.totalPnl());
            mtmStd[s].push_back(r.mtmStd);
            inv[s].push_back(r.avgEodInventory);
            hedge[s].push_back(r.hedgeCost);
            accept[s].push_back(100.0 * r.acceptRate());
            upliftOverall[s].push_back(pct(r.corePnl(), so.runs[TUNED].corePnl()));
            upliftSteady[s].push_back(pct(r.steadyPnl, so.runs[TUNED].steadyPnl));
        }
    }

    // ---- console report ----
    std::printf("%-30s %13s %13s %8s %13s %11s\n", "Strategy", "Core PnL ($)", "Steady PnL($)", "Accept%",
                "Avg |inv| ($)", "Hedge ($)");
    for (int s = 0; s < NUM_STRATEGIES; ++s)
        std::printf("%-30s %13.0f %13.0f %7.1f%% %13.0f %11.0f\n", STRATEGIES[s].label, mean(core[s]),
                    mean(steady[s]), mean(accept[s]), mean(inv[s]), mean(hedge[s]));

    std::printf("\nUplift vs tuned static rate sheet (mean +/- std over %d seeds):\n", opt.seeds);
    for (int s = 2; s < NUM_STRATEGIES; ++s)
        std::printf("  %-28s all days %+6.2f%% +/- %.2f   |   after warm-up %+6.2f%% +/- %.2f   [min %+.2f%%, max %+.2f%%]\n",
                    STRATEGIES[s].label, mean(upliftOverall[s]), stdev(upliftOverall[s]),
                    mean(upliftSteady[s]), stdev(upliftSteady[s]), minOf(upliftOverall[s]),
                    maxOf(upliftOverall[s]));
    std::printf("Uplift vs hand-set rate sheet:\n");
    for (int s = 2; s < NUM_STRATEGIES; ++s) {
        std::vector<double> u;
        for (const SeedOutcome& so : outcomes) u.push_back(pct(so.runs[s].corePnl(), so.runs[0].corePnl()));
        std::printf("  %-28s all days %+6.2f%% +/- %.2f\n", STRATEGIES[s].label, mean(u), stdev(u));
    }
    std::printf("Daily MTM risk (std-dev of daily MTM PnL, $): ");
    for (int s = 0; s < NUM_STRATEGIES; ++s) std::printf("%s=%.0f  ", STRATEGIES[s].id, mean(mtmStd[s]));
    std::printf("\n");

    // ---- JSON export ----
    JsonWriter j;
    j.beginObject();

    j.key("meta");
    j.beginObject();
    j.field("days", cfg.days);
    j.field("tradesPerDay", cfg.tradesPerDay);
    j.field("warmupDays", cfg.warmupDays);
    j.field("hardLimitUsd", cfg.hardLimitUsd);
    j.field("baseline", STRATEGIES[TUNED].id);
    j.key("seeds");
    {
        std::vector<int> seeds;
        for (const auto& so : outcomes) seeds.push_back(static_cast<int>(so.seed));
        j.array(seeds);
    }
    j.key("strategies");
    j.beginArray();
    for (const auto& s : STRATEGIES) {
        j.beginObject();
        j.field("id", s.id);
        j.field("label", s.label);
        j.endObject();
    }
    j.endArray();
    j.key("segments");
    {
        std::vector<std::string> names;
        for (const auto& s : SEGMENTS) names.push_back(s.name);
        j.array(names);
    }
    j.endObject();

    j.key("aggregate");
    j.beginObject();
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        j.key(STRATEGIES[s].id);
        j.beginObject();
        j.field("corePnlMean", mean(core[s]));
        j.field("corePnlStd", stdev(core[s]));
        j.field("steadyPnlMean", mean(steady[s]));
        j.field("totalPnlMean", mean(total[s]));
        j.field("acceptPct", mean(accept[s]));
        j.field("avgInventoryUsd", mean(inv[s]));
        j.field("hedgeCostMean", mean(hedge[s]));
        j.field("mtmStdMean", mean(mtmStd[s]));
        j.field("upliftPctMean", mean(upliftOverall[s]));
        j.field("upliftPctStd", stdev(upliftOverall[s]));
        j.field("upliftPctMin", minOf(upliftOverall[s]));
        j.field("upliftPctMax", maxOf(upliftOverall[s]));
        j.field("steadyUpliftPctMean", mean(upliftSteady[s]));
        j.field("steadyUpliftPctStd", stdev(upliftSteady[s]));
        j.key("upliftPctBySeed");
        j.array(upliftOverall[s]);
        j.key("steadyUpliftPctBySeed");
        j.array(upliftSteady[s]);
        j.endObject();
    }
    j.endObject();

    j.key("curves");
    j.beginObject();
    {
        std::vector<int> days;
        for (int d = 1; d <= cfg.days; ++d) days.push_back(d);
        j.key("day");
        j.array(days);
    }
    j.key("cumulativePnl");
    j.beginObject();
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::vector<double> daily = meanSeries(outcomes, s, &RunResult::dailyPnl);
        double run = 0.0;
        for (double& x : daily) {
            run += x;
            x = run;
        }
        j.key(STRATEGIES[s].id);
        j.array(daily);
    }
    j.endObject();
    j.key("inventoryUsd");
    j.beginObject();
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        j.key(STRATEGIES[s].id);
        j.array(meanSeries(outcomes, s, &RunResult::dailyInventory));
    }
    j.endObject();
    j.key("modelLoss");
    j.beginObject();
    j.key("learned");
    j.array(meanSeries(outcomes, 3, &RunResult::dailyModelLoss));
    j.key("bestPossible");
    j.array(meanSeries(outcomes, 3, &RunResult::dailyOracleLoss));
    j.endObject();
    j.endObject();

    j.key("segmentStats");
    j.beginObject();
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::vector<double> spread(NUM_SEGMENTS, 0.0), acc(NUM_SEGMENTS, 0.0);
        for (int g = 0; g < NUM_SEGMENTS; ++g) {
            std::vector<double> sp, ac;
            for (const auto& so : outcomes) {
                const RunResult& r = so.runs[s];
                if (r.segQuotes[g] > 0) {
                    sp.push_back(r.segSpreadSum[g] / r.segQuotes[g]);
                    ac.push_back(100.0 * r.segAccepts[g] / r.segQuotes[g]);
                }
            }
            spread[g] = mean(sp);
            acc[g] = mean(ac);
        }
        j.key(STRATEGIES[s].id);
        j.beginObject();
        j.key("avgSpreadBps");
        j.array(spread);
        j.key("acceptPct");
        j.array(acc);
        j.endObject();
    }
    j.endObject();

    j.key("acceptanceCurves");
    j.beginArray();
    for (const auto& c : curves) {
        j.beginObject();
        j.field("segment", c.segment);
        j.field("refBps", c.refBps);
        j.key("mult");
        j.array(c.mult);
        j.key("learned");
        j.array(c.learned);
        j.key("truth");
        j.array(c.truth);
        j.endObject();
    }
    j.endArray();

    j.key("learnedWeights");
    j.beginObject();
    j.key("names");
    j.array(weightNames);
    j.key("values");
    j.array(finalWeights);
    j.endObject();

    j.endObject();

    const std::string json = j.str();
    {
        std::ofstream f(opt.outDir + "/results.json");
        if (!f) {
            std::fprintf(stderr, "\ncould not write to '%s' - create the folder first\n", opt.outDir.c_str());
            return 1;
        }
        f << json << "\n";
    }
    {
        // A .js copy lets dashboard/index.html load the data straight from disk (no server needed).
        std::ofstream f(opt.outDir + "/data.js");
        f << "window.FXDATA = " << json << ";\n";
    }
    std::printf("\nWrote %s/results.json and %s/data.js\n", opt.outDir.c_str(), opt.outDir.c_str());
    return 0;
}

// test_core.cpp - quick sanity tests. Build and run with:  make test
#include <cmath>
#include <cstdio>
#include <string>

#include "../src/config.hpp"
#include "../src/engine.hpp"
#include "../src/market.hpp"
#include "../src/model.hpp"
#include "../src/rng.hpp"
#include "../src/strategy.hpp"

using namespace fx;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
            ++failures;                                               \
        } else {                                                      \
            std::printf("  ok:   %s\n", msg);                         \
        }                                                             \
    } while (0)

static void testRng() {
    std::printf("[rng]\n");
    Rng a(7), b(7), c(8);
    bool same = true, differs = false;
    for (int i = 0; i < 100; ++i) {
        const double x = a.normal(), y = b.normal(), z = c.normal();
        if (x != y) same = false;
        if (x != z) differs = true;
    }
    CHECK(same, "same seed -> identical stream");
    CHECK(differs, "different seed -> different stream");

    Rng r(1);
    double s = 0, s2 = 0;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
        const double x = r.normal();
        s += x;
        s2 += x * x;
    }
    CHECK(std::fabs(s / n) < 0.01, "normal mean is about 0");
    CHECK(std::fabs(s2 / n - 1.0) < 0.02, "normal variance is about 1");
}

static void testMarginalQuadrature() {
    std::printf("[market] Gauss-Hermite marginal acceptance vs Monte Carlo\n");
    Trade t;
    t.segment = HNI;
    t.pair = 1;
    t.side = -1;
    t.notional = 90000;
    t.loyalty = 0.3;
    t.vol = 1.4;
    Rng r(3);
    for (double spread : {20.0, 35.0, 50.0}) {
        double mc = 0;
        const int n = 300000;
        for (int i = 0; i < n; ++i) mc += acceptProb(t, spread, r.normal());
        mc /= n;
        const double gh = marginalAcceptProb(t, spread);
        CHECK(std::fabs(mc - gh) < 0.004, ("spread " + std::to_string(spread) + " bps: quadrature matches simulation").c_str());
    }
}

static void testStream() {
    std::printf("[market] stream generation\n");
    SimConfig cfg;
    const MarketStream a = generateStream(5, 10, 100, cfg);
    const MarketStream b = generateStream(5, 10, 100, cfg);
    CHECK(a.trades.size() == 1000, "days x tradesPerDay trades generated");
    bool same = true;
    for (std::size_t i = 0; i < a.trades.size(); ++i)
        if (a.trades[i].notional != b.trades[i].notional || a.trades[i].u != b.trades[i].u) same = false;
    CHECK(same, "same seed -> identical market stream");
    double share[NUM_SEGMENTS] = {0, 0, 0};
    const MarketStream big = generateStream(6, 40, 500, cfg);
    for (const Trade& t : big.trades) share[t.segment] += 1.0 / big.trades.size();
    CHECK(std::fabs(share[RETAIL] - 0.60) < 0.02 && std::fabs(share[CORPORATE] - 0.10) < 0.02,
          "segment mix matches configuration");
}

static void testModelLearnsTruth() {
    std::printf("[model] Newton/IRLS refit recovers the hidden price response\n");
    SimConfig cfg;
    const MarketStream train = generateStream(11, 40, 500, cfg);
    const MarketStream test = generateStream(12, 10, 500, cfg);
    Rng r(99);
    AcceptanceModel model(1.0);
    for (const Trade& t : train.trades) {  // random spreads around the rate sheet, like exploration
        const QuoteContext c = contextOf(t);
        const double spread = c.refBps * r.uniform(0.5, 1.6);
        FeatureVec x;
        buildFeatures(c, spread, x);
        model.addSample(x, t.u < acceptProb(t, spread, t.latent));
    }
    const int iters = model.refit();
    CHECK(iters > 0 && iters < 12, "refit converges before the iteration limit");

    double modelLoss = 0, oracleLoss = 0;
    for (const Trade& t : test.trades) {
        const QuoteContext c = contextOf(t);
        const double spread = c.refBps * r.uniform(0.5, 1.6);
        const bool y = t.u < acceptProb(t, spread, t.latent);
        FeatureVec x;
        buildFeatures(c, spread, x);
        modelLoss += logLoss(model.predict(x), y);
        oracleLoss += logLoss(marginalAcceptProb(t, spread), y);
    }
    modelLoss /= test.trades.size();
    oracleLoss /= test.trades.size();
    std::printf("        out-of-sample log-loss: model %.4f, best possible %.4f\n", modelLoss, oracleLoss);
    CHECK(modelLoss - oracleLoss < 0.01, "model is within 0.01 log-loss of the best possible predictor");
}

// A strategy that quotes a fixed spread, for accounting tests.
class FixedSpread : public Strategy {
public:
    explicit FixedSpread(double bps) : bps_(bps) {}
    std::string name() const override { return "fixed"; }
    double quote(const QuoteContext&, const Positions&, int) override { return bps_; }
private:
    double bps_;
};

static void testAccounting() {
    std::printf("[engine] accounting\n");
    SimConfig cfg;
    cfg.days = 20;
    const MarketStream m = generateStream(21, cfg.days, cfg.tradesPerDay, cfg);

    FixedSpread huge(100000.0);  // nobody accepts
    const RunResult none = simulate(m, huge, cfg);
    CHECK(none.accepts == 0 && none.corePnl() == 0.0, "absurd spreads -> no trades, zero PnL");

    FixedSpread zero(0.0);  // everybody with reservation > 0 accepts, but we earn nothing
    const RunResult free = simulate(m, zero, cfg);
    CHECK(free.spreadRevenue == 0.0 && free.hedgeCost > 0.0, "zero spread -> no revenue, positive hedge cost");
    CHECK(free.corePnl() < 0.0, "giving away FX at mid loses money");

    StaticStrategy sheet("sheet", handSetRateSheet());
    const RunResult r = simulate(m, sheet, cfg);
    CHECK(r.accepts > 0 && r.accepts <= r.quotes, "accepted trades <= quotes");
    CHECK(r.quotes == static_cast<long>(cfg.days) * cfg.tradesPerDay, "every quote request is counted");
    CHECK(r.corePnl() > 0.0, "the hand-set rate sheet is profitable");
}

static void testDynamicBeatsSheet() {
    std::printf("[strategy] dynamic pricing vs rate sheet (one fresh seed)\n");
    SimConfig cfg;
    const MarketStream m = generateStream(42, cfg.days, cfg.tradesPerDay, cfg);
    StaticStrategy sheet("sheet", handSetRateSheet());
    DynamicParams p;
    DynamicStrategy dyn("dynamic", p, cfg.warmupDays, 42);
    const RunResult a = simulate(m, sheet, cfg);
    const RunResult b = simulate(m, dyn, cfg);
    std::printf("        steady-state PnL: sheet %.0f, dynamic %.0f\n", a.steadyPnl, b.steadyPnl);
    CHECK(b.steadyPnl > a.steadyPnl, "after warm-up the dynamic engine out-earns the hand-set sheet");
}

int main() {
    testRng();
    testMarginalQuadrature();
    testStream();
    testModelLearnsTruth();
    testAccounting();
    testDynamicBeatsSheet();
    std::printf("\n%s (%d failure%s)\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

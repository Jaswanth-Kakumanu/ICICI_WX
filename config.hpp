// config.hpp - every assumption of the simulated FX business lives here.
//
// Spreads are quoted in basis points (bps) of the mid rate, 1 bp = 0.01%.
// Notionals are in USD-equivalent. All numbers are ILLUSTRATIVE assumptions for
// a simulation, not real ICICI Bank data.
#pragma once

namespace fx {

constexpr int NUM_SEGMENTS = 3;
constexpr int NUM_PAIRS = 5;

enum Segment : int { RETAIL = 0, HNI = 1, CORPORATE = 2 };

// Hidden "ground truth" of how each customer segment reacts to price.
// A customer accepts a quote with probability sigmoid((reservation - spread) / width).
struct SegmentSpec {
    const char* name;
    double share;           // share of incoming quote requests
    double medianNotional;  // USD, median trade size
    double sigmaNotional;   // log-normal sigma of trade size
    double reservationBps;  // typical maximum spread the customer tolerates
    double widthBps;        // how sharply acceptance falls with spread (small = sharp)
    double sizeElasticity;  // bigger trades -> customers tolerate less spread
    double heterogeneity;   // sigma of the unobservable customer-specific shock (log scale)
    double sheetBps;        // spread on the hand-set rate sheet
};

static constexpr SegmentSpec SEGMENTS[NUM_SEGMENTS] = {
    //  name         share  median   sigma  resv   width  elast  hetero  sheet
    {"Retail",       0.60,    3000.0, 0.80, 110.0, 18.0,  0.15,  0.20,   85.0},
    {"HNI",          0.30,   60000.0, 0.70,  35.0,  6.0,  0.25,  0.20,   27.0},
    {"Corporate",    0.10,  600000.0, 0.80,   9.0,  1.6,  0.35,  0.20,    7.0},
};

struct PairSpec {
    const char* name;
    double share;            // share of requests
    double toleranceFactor;  // scales spreads for this pair (exotic pairs are wider)
    double hedgeBps;         // cost to unwind inventory in the interbank market (at vol = 1)
    double dailyVol;         // daily return volatility (fraction, at vol = 1)
};

static constexpr PairSpec PAIRS[NUM_PAIRS] = {
    //  name       share  tol   hedge  dailyVol
    {"USD/INR",    0.45, 1.00,  1.5,  0.0030},
    {"EUR/INR",    0.20, 1.20,  3.0,  0.0050},
    {"GBP/INR",    0.12, 1.30,  4.0,  0.0055},
    {"AED/INR",    0.13, 0.90,  2.0,  0.0030},
    {"SGD/INR",    0.10, 1.25,  4.5,  0.0045},
};

// Settings of the simulated world.
struct SimConfig {
    int days = 60;
    int tradesPerDay = 500;      // quote requests per day across all pairs
    int warmupDays = 8;          // dynamic engine explores randomly during these days
    double hardLimitUsd = 4'000'000.0;  // per-pair position limit; breaching it forces a hedge
    double hedgeBackFraction = 0.5;     // a forced hedge brings the position back to limit * this
    double flowBiasSigma = 0.12;        // daily one-sidedness of customer flow per pair
};

// Settings of the dynamic pricing engine.
struct DynamicParams {
    bool inventoryAware = false;

    // Candidate spreads are multiples of the pair/segment reference spread.
    double minMult = 0.35;
    double maxMult = 2.00;
    int gridSteps = 166;  // (maxMult - minMult) / 0.01 + 1

    // Logistic regression, refitted every night. Ridge = strength of the L2 penalty.
    double ridge = 1.0;

    // Exploration.
    double warmupLo = 0.6, warmupHi = 1.5;    // random multiplier range during warm-up days
    double epsilon = 0.03;                    // afterwards: share of quotes that explore
    double exploreLo = 0.7, exploreHi = 1.4;

    // Inventory-aware pricing.
    double nettingTheta = 0.6;   // share of new one-sided flow expected to end up hedged
    double riskAversion = 3e-6;  // price of variance of the open position (per USD)
};

}  // namespace fx

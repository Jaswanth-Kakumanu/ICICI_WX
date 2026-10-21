# Dynamic FX Pricing Engine

A C++17 simulation of a customer-centric FX pricing desk. It simulates customers who accept or reject FX quotes, learns how each customer responds to price, then quotes the spread that maximises expected profit for that customer and trade, while managing inventory. Results are compared against static rate sheets and shown in a web dashboard.

Everything is simulated. No real bank data is used, and every assumption is in `src/config.hpp`.

## Quick start

```bash
make test     # about 20 sanity checks
make run      # runs 10 random markets, writes dashboard/data.js and dashboard/results.json
# then open dashboard/index.html in a browser (no server needed)
```

Needs only `g++` (C++17) and `make`. Without make:
`g++ -std=c++17 -O2 src/main.cpp -o build/fxprice && ./build/fxprice --seeds 10 --out dashboard`

Options: `--seeds N --days D --trades T --out DIR --gamma G --theta T --first-seed S`.
A full 10-seed run takes a few seconds.

## Project layout

| File | What it does |
|---|---|
| `src/config.hpp` | Segments, currency pairs, hidden customer behaviour, sim and model settings |
| `src/rng.hpp` | Deterministic random numbers (same seed = same result on any machine) |
| `src/market.hpp` | Customer simulator, ground-truth acceptance behaviour, static rate sheets |
| `src/model.hpp` | Feature engineering and logistic regression written from scratch (Newton/IRLS) |
| `src/strategy.hpp` | `StaticStrategy` and `DynamicStrategy` (grid-search spread optimiser, inventory penalty) |
| `src/engine.hpp` | Replays a market against a strategy: PnL, hedging, inventory, diagnostics |
| `src/json.hpp` | Tiny JSON writer, no external libraries |
| `src/main.cpp` | Multi-seed experiment, console report, dashboard export |
| `tests/test_core.cpp` | Unit and sanity tests |
| `dashboard/index.html` | Results dashboard, plain HTML/CSS/JS with hand-drawn SVG charts |

## How it works

**1. Simulated world.** Each day has 500 quote requests. A request has a segment (Retail, HNI, Corporate), a currency pair (USD, EUR, GBP, AED, SGD against INR), a side, a trade size, a loyalty score and a market volatility level. Customer flow is randomly one-sided each day, so the bank builds up inventory.

**2. Hidden customer behaviour.** A customer accepts a quote with probability

```
P(accept) = sigmoid( (reservation - spread) / width )
```

The reservation spread depends on segment, pair, trade size, loyalty, volatility, side, plus a hidden personal shock the bank can never observe. The engine has to discover all this from accept/reject outcomes alone.

**3. Baselines.** *Hand-set sheet:* one spread per segment and pair. *Tuned sheet:* for each segment and pair, the best fixed spread found on a separate, longer history (like a desk running A/B tests). The tuned sheet is the baseline used for headline numbers because it is hard to beat.

**4. Learning.** Every quote and its outcome is stored. Each night a logistic regression (30 features) is refitted on all data with Newton's method (IRLS). The first 8 days use randomised quotes so the model sees customers react to many prices, and 3% of later quotes keep exploring.

**5. Optimising the quote.** For each request the engine tries 166 candidate spreads and picks the one maximising

```
P(accept | spread) x (revenue - inventory cost)
```

**6. Inventory awareness.** If the bank is already short USD, a customer buying more USD makes things worse and one selling USD makes things better. The inventory cost is the expected hedging cost of the position change plus a variance (risk) penalty, so the engine skews prices to attract offsetting flow. Positions above a hard limit are hedged at a cost, and everything is flattened at day end.

**7. Fair evaluation.** All strategies replay the *same* customers (common random numbers). The tuned sheet is fitted on different random seeds than it is tested on. The inventory penalty parameters (`gamma`, `theta`) were chosen on seeds 101-104 and results are reported on seeds 1-10. Profit ("core PnL") is spread revenue minus hedging cost; mark-to-market noise is reported separately as risk.

## Results from the included run (10 seeds, 60 days x 500 quotes)

| Strategy | Profit / market | vs tuned sheet | Avg open position | Hedging cost |
|---|---|---|---|---|
| Hand-set sheet | $2.40M | -1.2% | $7.3M | $171k |
| **Tuned sheet (baseline)** | $2.42M | - | $7.7M | $198k |
| Dynamic pricing | $2.55M | +5.1% | $7.6M | $202k |
| Dynamic + inventory-aware | $2.55M | **+5.2% +/- 1.0** | $6.5M | $145k |

- After the 8-day learning period the uplift is **+7.8% +/- 0.9**. Every one of the 10 markets improved (range +3.7% to +7.0% over all days).
- Inventory awareness leaves profit unchanged but cuts hedging cost by 28%, average open position by 15% and daily price risk by 18% versus dynamic pricing without it.
- The learned price-response curves match the hidden truth closely, and model log-loss reaches the theoretical floor after a few days.

**These numbers belong to this simulator and its assumptions.** They show the method works in a controlled world, not what a real desk would earn. Change the assumptions and the numbers change. Always quote your own run.

## Resume bullet

> Built **[project name]**, a C++17 FX pricing engine with a customer simulator, from-scratch logistic-regression acceptance model (Newton/IRLS) and inventory-aware spread optimiser; improved simulated profit by **5.2% (7.8% after warm-up)** over a tuned static rate-sheet baseline across **10** random seeds while cutting hedging cost by **28%**; visualised results in an HTML/JS dashboard.

Re-run with your own settings and use your own numbers. Keep the word "simulated".

## Questions you should be ready for

- **Why logistic regression, not XGBoost?** The outcome is binary and I needed calibrated probabilities that I can plug straight into an expected-profit formula. Logistic regression gives that, is fast to refit, and I could implement and debug it myself. It also matched the best-possible log-loss here.
- **What is IRLS / Newton's method?** Each step solves `H * step = gradient` where `H` is the Hessian of the log-loss. Warm-started from yesterday's weights it converges in 1-3 iterations. I used it because plain SGD converged too slowly and gave an under-sensitive model that over-quoted spreads.
- **Why explore?** Without varied prices in the data the model cannot learn how acceptance changes with spread. Warm-up plus 3% epsilon exploration costs a little profit (see the first days on the dashboard) and buys the model.
- **Why is the baseline "tuned"?** A hand-set sheet is easy to beat. Beating a per-cell tuned sheet on unseen seeds is a fairer test.
- **Where could it be wrong?** Customer behaviour is my assumption. A real deployment would need real quote/acceptance history, competitor prices, regulatory and fairness checks, and a live A/B test.
- **What would you do next?** Add per-customer history, model uncertainty (explore where the model is unsure), a gradient-boosted model to compare, and a live quote API in front of the engine.

## Ideas to extend

- Add a second learner (e.g. plain SGD) behind the same interface and compare.
- Make customers repeat over time, so loyalty is earned rather than random.
- Add competitor quotes as a feature.
- Serve the engine as a small HTTP service and call it from the dashboard.

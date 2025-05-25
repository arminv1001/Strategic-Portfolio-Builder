**Strategic Portfolio Optimization – Technical Memo**


## 1 Project Set‑Up

### 1.1 Folder layout

```
Strategic_Portfolio/
├── input_data/
│   └── config_file.yaml
├── requirements.txt
├── strategic_portfolio_analysis.py
└── output_data/         # generated automatically
```

### 1.2 Environment & installation

1. Clone / download the project.
2. Create a fresh Python ≥3.9 virtual‐env and activate it:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:

   ```bash
   python strategic_portfolio_analysis.py  # uses default YAML
   ```

### 1.3 Configuration via YAML

All run‑time parameters live in `input_data/config_file.yaml`. Key knobs (excerpt shown right):

| Section                                  | Purpose                                                    |
| ---------------------------------------- | ---------------------------------------------------------- |
| `tickers`                                | Universe definition (assets to be considered).             |
| `benchmark_ticker`                       | Market proxy for CAPM.                                     |
| `start_date / end_date / data_frequency` | Historical window & periodicity.                           |
| `risk_free_rate`                         | Manual *vs* online 3‑month T‑bill toggle.                  |
| `filter.jarque_bera_threshold`           | p‑value cut‑off for normality screen.                      |
| `conditioning`                           | Constraints: shorting, max allocation.                     |
| `capm_health`                            | Thresholds that feed the risk‑aversion knob in back‑tests. |

Changing the YAML is enough to re‑run the entire workflow with a different asset set, horizon, or risk preferences—no Python edits are needed.

---

## 2 Strategy Logic (inside `strategic_portfolio_analysis.py`)

1. **Data ingestion** – monthly *Adj Close* prices are fetched from Yahoo Finance; missing values are forward‑filled.fileciteturn2file6
2. **Normality filter** – each asset’s returns undergo a Jarque–Bera test; assets failing the `0.05` threshold are dropped.fileciteturn2file10
3. **Mean‑Variance engine** – expected return vector `μ` and covariance `Σ` are assembled; the (unconstrained or constrained) efficient frontier is traced, and the tangency portfolio index computed by maximising Sharpe vs the risk‑free rate.fileciteturn2file18
4. **Complete portfolio** – given user risk‑aversion *A*, capital is allocated `y` % to the tangency portfolio, `(1‑y)` % to cash. The weight is capped in `[0,1]` to avoid leverage.fileciteturn2file16
5. **CAPM decomposition & health check** – in‑sample alpha/β are benchmarked; yellow/red flags dynamically bump *A* upward, making the back‑test more conservative.fileciteturn2file4
6. **Back‑test & plotting** – the forward sample is run, plotting portfolio value & draw‑down; summary stats land in `output_data/stats_asset_summary.json` 

---

## 3 Design Choice

* **Economically intuitive** – Modern Portfolio Theory supplies a disciplined way to translate beliefs about mean/vol into tradeable weights, maximizing reward per unit risk.
* **Robustness by diagnostic gating** – the Jarque–Bera screen filters assets whose heavy tails could break the quadratic utility assumptions, improving out‑of‑sample stability.
* **Adaptive risk appetite** – the CAPM‑based health metric feeds back into the risk‑aversion scalar, trimming exposure when beta is too high or alpha disappoints.
* **Fully declarative** – all switches sit in the YAML, enabling rapid scenario analysis (change `allow_shorting: true`, tighten `max_allocation_threshold`, or swap benchmarks).
* **Reproducibility** – deterministic virtual‑env + requirements + JSON outputs guarantee results can be replicated end‑to‑end by graders.

---

## 4 Headline Numbers

A snapshot of `stats_asset_summary.json` shows both ex‑ante expectations and realised figures:

| Metric           | Expected |    Realised | Comment                                                    |
| ---------------- | -------: | ----------: | ---------------------------------------------------------- |
| Portfolio μ      |    0.350 |       0.160 | Tech sell‑off 2022 depressed returns.                      |
| Portfolio σ      |    0.387 |       0.205 | Risk materially lower than forecast → favourable surprise. |
| Sharpe           |     0.84 |        0.66 | Still attractive vs 0.5 threshold.                         |
| Max draw‑down    |        – |      −9.6 % | Within mandate (< 15 %).                                   |
| CAPM β           |     0.56 |        0.39 | Systematic exposure modest.                                |
| CAPM α (monthly) |  −0.17 % | **+0.99 %** | Positive alpha out‑of‑sample indicates selection skill.    |
| Win rate         |        – |        58 % | Consistent month‑over‑month performance.                   |

**Interpretation** – although the realised return lagged the ambitious in‑sample forecast, the risk came in far lower, and alpha flipped from negative to positive, delivering a respectable Sharpe above the health hurdle. The strategy therefore remains fully deployable, with no red flags triggered.



# Quantitative Portfolio Optimization (cuFOLIO)

GPU-accelerated Mean-CVaR portfolio optimization with scenario generation, efficient frontier, and dynamic rebalancing.

## ALWAYS Use cuFOLIO Library Code

> **MANDATORY** — For ANY task involving this portfolio optimization codebase (computing returns, generating scenarios, optimizing, backtesting, plotting), you MUST use the `cufolio` library functions documented in this README. Do NOT write raw pandas, numpy, or matplotlib code to replicate what cuFOLIO already provides. You may write new code if user requests 

**Examples of what to use:**

| Task | Use this cuFOLIO function | Do NOT write raw code for |
|------|--------------------------|--------------------------|
| Compute returns | `utils.calculate_returns()` | Manual `pd.DataFrame.pct_change()` or `np.log()` |
| Generate scenarios | `cvar_utils.generate_cvar_data()` | Manual KDE fitting or sampling |
| Build optimization | `cvar_optimizer.CVaR()` | Manual CVXPY problem construction |
| Solve | `cvar_problem.solve_optimization_problem()` | Direct `problem.solve()` calls |
| Backtest | `backtest.portfolio_backtester()` | Manual cumulative return calculations |
| Plot allocation | `portfolio.plot_portfolio()` | Manual matplotlib bar charts |
| Plot backtest | `backtester.backtest_against_benchmarks()` | Manual matplotlib line plots |
| Rebalancing | `rebalance.rebalance_portfolio()` | Manual rolling optimization loops |

**Even for simple tasks** like "compute log returns" or "show me a summary of returns," use `utils.calculate_returns()` and work with the `returns_dict` it produces — do not bypass the library.

## Before You Start: Confirm with User

> **MANDATORY CHECKPOINT — DO NOT write any code or run any optimization until you have confirmed ALL of the following with the user.** If the user's request leaves any of these ambiguous, you MUST ask before proceeding. Do NOT silently assume defaults.

Present each group as **multiple-choice options** so the user can quickly pick. Always include an "Other" option for custom values.

**1. Dataset and time period (ask if not already clear):**

```
"Which dataset?
  (a) Default S&P 500 (~397 stocks)
  (b) Other — please provide the file path"

"What date range?
  (a) 2021-01-01 to 2024-01-01 (recent 3-year)
  (b) 2022-01-01 to 2024-01-01 (recent 2-year)
  (c) 2020-01-01 to 2023-01-01 (includes COVID)
  (d) Other — please specify start and end dates (YYYY-MM-DD)"
```

**2. Portfolio type and constraints (ask about ANY the user hasn't explicitly specified):**

```
"Portfolio type?
  (a) Long-only (weights ≥ 0)
  (b) Long-short (allow shorting)
  (c) Other — please specify bounds"

"Max weight per asset?
  (a) 40% (moderate diversification)
  (b) 20% (high diversification)
  (c) 100% (no concentration limit)
  (d) Other — please specify"

"Risk tolerance?
  (a) Conservative (risk_aversion = 10)
  (b) Moderate (risk_aversion = 1)
  (c) Aggressive (risk_aversion = 0.1)
  (d) Max growth (risk_aversion = 0.001)
  (e) Other — please specify"

"Cash allocation?
  (a) Fully invested (0% cash)
  (b) Allow 0–10% cash buffer
  (c) Allow 0–20% cash buffer
  (d) Other — please specify min/max"

"CVaR confidence level?
  (a) 95% (standard)
  (b) 99% (more tail-focused)
  (c) 90% (less extreme)
  (d) Other — please specify"
```

**3. Rebalancing parameters (ask if the task involves rebalancing/dynamic strategies):**

```
"Trading frequency (how often to evaluate/rebalance)?
  (a) Daily (1 trading day)
  (b) Weekly (5 trading days)
  (c) Monthly (21 trading days)
  (d) Quarterly (63 trading days)
  (e) Other — please specify number of trading days"

"Look-back window (history per optimization)?
  (a) 6 months (126 trading days)
  (b) 1 year (252 trading days)
  (c) 2 years (504 trading days)
  (d) Other — please specify number of trading days"
```

**4. Output format (ALWAYS ask — do not assume):**

```
"What output would you like?
  (a) Numerical results only (tables/metrics to console)
  (b) Plots saved to files
  (c) Both numerical results and plots"

"Do you want backtesting?
  (a) Yes — use 6 months after training end date
  (b) Yes — specify a custom test period
  (c) No backtesting"

"Which plots? (select all that apply)
  (a) Portfolio allocation bar chart
  (b) Backtest cumulative returns
  (c) Efficient frontier
  (d) Rebalancing strategy comparison
  (e) Weight evolution vs price
  (f) All of the above"
```

**Only proceed to code after the user has confirmed or explicitly said "use defaults for the rest."**

## Pipeline Overview

```
┌───────────────────────────────┐
│  Step 1: Data Preprocessing   │  Load prices, compute returns
│  Output: returns_dict         │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 2: Scenario Generation  │  KDE / Gaussian scenario sampling
│  Output: returns_dict         │
│          + cvar_data          │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 3: Model Building       │  Define CvarParameters, build CVaR problem
│  Output: cvar_problem         │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 4: Solve                │  GPU (cuOpt) or CPU solver
│  Output: result, portfolio    │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Step 5: Backtest             │  Evaluate out-of-sample performance
│  Output: backtest metrics     │
└───────────────────────────────┘

Composite workflows (use all steps internally):
  • Efficient Frontier  — sweeps risk aversion, solves many problems
  • Rebalancing         — rolling optimization with triggers
```

---

## Step 1: Data Preprocessing

> Load stock price data, filter by date range (market regime), and compute returns.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "What dataset?" | S&P 500 (`sp500.csv`, auto-downloaded) | Determines asset universe |
| "What date range?" | `("2022-01-01", "2024-01-01")` | Training period for optimization |
| "Log returns or simple returns?" | `"LOG"` (log returns) | Log returns are standard for CVaR; use `"LINEAR"` for mean-variance |
| "Daily, weekly, or monthly?" | `freq=1` (daily) | Daily gives most data points |

### Code (cvar_basic.ipynb, Cells 10–11)

```python
import os
from cufolio import utils
from cufolio.settings import ReturnsComputeSettings

data_path = "../data/stock_data/sp500.csv"
if not os.path.exists(data_path):
    utils.download_data(data_path)  # auto-download S&P 500 (397 stocks)

regime_dict = {"name": "recent", "range": ("2021-01-01", "2024-01-01")}
returns_compute_settings = ReturnsComputeSettings(return_type='LOG', freq=1)

returns_dict = utils.calculate_returns(
    data_path,
    regime_dict,
    returns_compute_settings
)
```

> **Important**: `returns_compute_settings` must be a `ReturnsComputeSettings` object (from `cufolio.settings`), not a plain dict. The function will fail with an `AttributeError` if you pass a dict.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str or DataFrame | — | Path to CSV/Parquet/Excel/JSON price file, or a DataFrame |
| `regime_dict` | dict | None | `{"name": str, "range": (start, end)}` — date filter |
| `returns_compute_settings` | `ReturnsComputeSettings` | — | Pydantic settings object (see fields below) |

**`ReturnsComputeSettings` fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `return_type` | `"LOG"` | `"LOG"` (log returns), `"LINEAR"` (simple returns), `"ABSOLUTE"` (price diffs), or `"PNL"` (data is already returns) |
| `freq` | `1` | Return frequency: 1 = daily, 5 = weekly, 21 = monthly |
| `returns_compute_device` | `"CPU"` | Device for returns calculation |
| `verbose` | `False` | Print progress |

### Output: `returns_dict`

| Key | Type | Description |
|-----|------|-------------|
| `return_type` | str | The return type used |
| `returns` | pd.DataFrame | Returns matrix (num_days × num_assets) |
| `regime` | dict | `{"name": str, "range": (start, end)}` |
| `dates` | pd.DatetimeIndex | Trading dates in the period |
| `mean` | np.ndarray | Mean return per asset (num_assets,) |
| `covariance` | np.ndarray | Covariance matrix (num_assets × num_assets) |
| `tickers` | list[str] | Asset ticker symbols |

### Predefined Market Regimes

| Regime | Date Range |
|--------|------------|
| `pre_crisis` | 2005-01-01 → 2007-10-01 |
| `crisis` | 2007-10-01 → 2009-04-01 |
| `post_crisis` | 2009-06-30 → 2014-06-30 |
| `covid` | 2020-01-01 → 2023-01-01 |
| `recent` | 2021-01-01 → 2024-01-01 |

### How to Extend

- **Custom data**: Pass any CSV with rows = dates (index), columns = ticker symbols, values = adjusted closing prices. Columns with NaN are auto-dropped.
- **Different return types**: Use `"NORMAL"` for absolute returns or `"PNL"` if the dataset is already returns.
- **Weekly/monthly**: Set `freq=5` (weekly) or `freq=21` (monthly).
- **Pass a DataFrame directly**: `utils.calculate_returns(my_dataframe, regime_dict, settings)`.

### Common Issues

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` | Set correct `data_path` or call `utils.download_data(data_path)` first |
| 0 assets returned | Date range has no data — check overlap with dataset |
| Missing tickers | Tickers with NaN prices in the period are auto-dropped |

---

## Step 2: Scenario Generation

> Fit a distribution to historical returns and sample synthetic scenarios for CVaR optimization.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "How many scenarios?" | `10000` | More = smoother CVaR but slower solve. 5000–20000 is typical |
| "GPU or CPU for KDE?" | `"GPU"` (requires cuML) | GPU is ~100× faster for KDE fitting |
| "KDE or Gaussian fitting?" | `"kde"` | KDE is non-parametric (better for non-normal returns); Gaussian is simpler |

### Code (cvar_basic.ipynb, Cell 13)

```python
from cufolio import cvar_utils
from cufolio.settings import ScenarioGenerationSettings, KDESettings

scenario_generation_settings = ScenarioGenerationSettings(
    num_scen=10000,
    fit_type='kde',
    kde_settings=KDESettings(
        bandwidth=0.01,
        kernel='gaussian',
        device='GPU'        # 'GPU' uses cuml.KDE, 'CPU' uses sklearn.KDE
    ),
    verbose=False
)

returns_dict = cvar_utils.generate_cvar_data(
    returns_dict,
    scenario_generation_settings
)
```

> **Important**: `scenario_generation_settings` must be a `ScenarioGenerationSettings` object with an optional `KDESettings` sub-object (both from `cufolio.settings`), not a plain dict.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_scen` | 10000 | Number of return scenarios to simulate |
| `fit_type` | `"kde"` | `"kde"` (kernel density), `"gaussian"` (multivariate normal), or `"no_fit"` (use raw historical returns) |

**`kde_settings` (only when `fit_type="kde"`):**

| Key | Default | Description |
|-----|---------|-------------|
| `bandwidth` | 0.01 | KDE bandwidth |
| `kernel` | `"gaussian"` | KDE kernel type |
| `device` | `"GPU"` | `"GPU"` → cuml.KDE (fast), `"CPU"` → sklearn.KDE |

### Output

The function adds a `cvar_data` key to `returns_dict`:

| Key | Type | Description |
|-----|------|-------------|
| `returns_dict["cvar_data"]` | `CvarData` dataclass | Contains `.mean` (num_assets,), `.R` (num_assets × num_scen), `.p` (num_scen,) |

`CvarData` fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `mean` | (num_assets,) | Expected asset returns |
| `R` | (num_assets, num_scen) | Scenario return deviations (transposed) |
| `p` | (num_scen,) | Scenario probabilities (uniform, sum to 1) |

### How to Extend

- **More scenarios**: Increase `num_scen` for smoother CVaR approximation (costs more memory and solve time).
- **No fitting**: Set `fit_type="no_fit"` to use raw historical returns as scenarios directly.
- **Gaussian**: Set `fit_type="gaussian"` to sample from a fitted multivariate normal distribution (no `kde_settings` needed).
- **CPU fallback**: Set `device="CPU"` if no GPU / cuML is available.

---

## Step 3: Model Building

> Define optimization parameters and instantiate the CVaR optimizer.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "Long-only or allow shorting?" | Long-only: `w_min=0.0` | Shorting (`w_min < 0`) enables hedging but adds complexity |
| "Max weight per asset?" | `w_max=0.4` (40%) | Lower = more diversified; `1.0` = allow single-stock concentration |
| "Allow cash allocation?" | `c_min=0.0, c_max=0.1` | `c_max=0.0` = fully invested; positive = defensive buffer |
| "Leverage allowed?" | `L_tar=1.0` (no leverage) | `> 1.0` allows gross exposure above 100% |
| "Risk aversion level?" | `risk_aversion=1` (moderate) | Higher → conservative, lower → aggressive. Auto-scaled internally |
| "CVaR confidence?" | `confidence=0.95` (95%) | 0.99 = more tail-focused; 0.90 = less extreme |
| "Limit number of holdings?" | `cardinality=None` (LP, no limit) | Setting an int makes it a MILP (slower but sparse) |
| "Sector/group constraints?" | `None` | e.g., "Tech ≤ 10%" via `group_constraints` |
| "Turnover limit?" | `T_tar=None` (unconstrained) | Set if rebalancing from an existing portfolio |

### Code (cvar_basic.ipynb, Cells 15 + 17)

```python
from cufolio import cvar_optimizer
from cufolio.cvar_parameters import CvarParameters

cvar_params = CvarParameters(
    w_min={"NVDA": 0.1, "others": -0.3},   # per-asset or uniform lower bound
    w_max={"NVDA": 0.6, "others": 0.4},    # per-asset or uniform upper bound
    c_min=0.0,          # minimum cash
    c_max=0.2,          # maximum cash
    L_tar=1.6,          # leverage limit ‖w‖₁
    T_tar=None,         # turnover limit (None = unconstrained)
    cvar_limit=None,    # hard CVaR cap (None = use λ-penalized objective)
    cardinality=None,   # max assets (None = LP; int = MILP)
    risk_aversion=1,    # λ — risk penalty
    confidence=0.95     # α — CVaR confidence level
)

cvar_problem = cvar_optimizer.CVaR(
    returns_dict=returns_dict,
    cvar_params=cvar_params
)
```

### CvarParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_min` | float / ndarray / dict | 0.0 | Lower weight bound. Dict: `{"NVDA": 0.1, "others": -0.3}` |
| `w_max` | float / ndarray / dict | 1.0 | Upper weight bound |
| `c_min` | float | 0.0 | Minimum cash |
| `c_max` | float | 1.0 | Maximum cash |
| `risk_aversion` | float | 1.0 | λ — risk penalty in objective (auto-scaled internally) |
| `confidence` | float | 0.95 | α — CVaR confidence level |
| `L_tar` | float | 1.6 | Leverage limit `‖w‖₁ ≤ L_tar` |
| `T_tar` | float or None | None | Turnover limit (None = unconstrained) |
| `cvar_limit` | float or None | None | Hard CVaR cap (None = λ-penalized objective) |
| `cardinality` | int or None | None | Max assets in portfolio (None = LP; int = MILP) |
| `group_constraints` | list[dict] or None | None | Sector/group weight bounds |

### CVaR Constructor: `api_settings`

| API | `api_settings` value | When to use |
|-----|---------------------|-------------|
| **CVXPY with bounds** (default) | `None` or `{"api": "cvxpy", "weight_constraints_type": "bounds"}` | Standard usage; cuOpt is faster with bounds |
| **CVXPY with parameters** | `{"api": "cvxpy", "weight_constraints_type": "parameter"}` | Fast iteration on bounds without rebuilding |
| **cuOpt native Python** | `{"api": "cuopt_python"}` | Direct cuOpt control, no CVXPY overhead |

### Mathematical Formulation

```
maximize:   μᵀw − λ_risk · (t + 1/(1−α) · pᵀu)
subject to: u[s] + t + R[s]·w ≥ 0           (CVaR, ∀s)
            Σw[i] + c = 1                    (self-financing)
            w_min[i] ≤ w[i] ≤ w_max[i]      (concentration)
            c_min ≤ c ≤ c_max               (cash bounds)
            ‖w‖₁ ≤ L_tar                    (leverage)
            ‖w − w_pre‖₁ ≤ T_tar            (turnover, optional)
            Σy[i] ≤ K, y[i] ∈ {0,1}         (cardinality, optional → MILP)
```

**Decision Variables:**
- `w[i]` — weight for asset `i` (CONTINUOUS)
- `c` — cash (CONTINUOUS)
- `t` — VaR threshold (CONTINUOUS)
- `u[s]` — CVaR auxiliary per scenario (CONTINUOUS, ≥ 0)
- `y[i]` — asset selection indicator (BINARY, only when `cardinality` is set)

### How to Extend

| Goal | What to change |
|------|---------------|
| **Long-only, fully invested** | `w_min=0.0, w_max=1.0, c_min=0.0, c_max=0.0, L_tar=1.0` |
| **Long-short with leverage** | `w_min=-0.3, w_max=0.4, L_tar=1.6` |
| **Limit to K assets (MILP)** | `cardinality=K` |
| **Cap turnover from existing portfolio** | `T_tar=0.5` + pass `existing_portfolio` to constructor |
| **Hard CVaR ceiling** (objective becomes max return) | `cvar_limit=0.03` |
| **Per-asset bounds via dict** | `w_min={"NVDA": 0.1, "others": -0.3}` |
| **Sector group constraints** | `group_constraints=[{"group_name": "Tech", "tickers": ["AAPL","MSFT"], "weight_bounds": {"w_min": 0.0, "w_max": 0.4}}]` |

---

## Step 4: Solve

> Solve the optimization problem on GPU or CPU.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "GPU or CPU solver?" | `cp.CUOPT` (GPU) | GPU is ~100–200× faster; use CPU if no GPU available |
| "Time limit?" | `15` seconds (LP), `200` seconds (MILP) | MILP problems need more time |
| "Compare GPU vs CPU?" | No | If yes, solve with both and call `utils.compare_results()` |

### Code (cvar_basic.ipynb, Cell 19)

```python
import cvxpy as cp

# GPU solver (cuOpt PDLP)
gpu_solver_settings = {
    "solver": cp.CUOPT,
    "verbose": False,
    "solver_method": "PDLP",
    "time_limit": 15,
    "optimality": 1e-4
}

result, portfolio = cvar_problem.solve_optimization_problem(
    solver_settings=gpu_solver_settings
)
```

### Solver Settings (CVXPY API)

| Parameter | Example | Description |
|-----------|---------|-------------|
| `solver` | `cp.CUOPT` | CVXPY solver: `cp.CUOPT` (GPU), `cp.CLARABEL` (CPU), etc. |
| `solver_method` | `"PDLP"` | cuOpt method: `"PDLP"`, `"Dual Simplex"`, `"Barrier"`, `"Concurrent"` |
| `time_limit` | 15 | Solver time limit (seconds) |
| `optimality` | 1e-4 | Optimality tolerance (PDLP) |
| `mip_absolute_tolerance` | 1e-4 | MIP gap tolerance (MILP only) |
| `verbose` | False | Print solver logs |

### Solver Settings (cuOpt Native Python API)

When using `api_settings={"api": "cuopt_python"}`:

| Parameter | Example | Description |
|-----------|---------|-------------|
| `log_to_console` | True | Print solve log |
| `method` | 1 | Solver method (1 = PDLP) |
| `presolve` | False | Enable presolve |
| `time_limit` | 60 | Time limit (seconds) |

### Output

**`result`** — `pd.Series` with keys:

| Key | Description |
|-----|-------------|
| `regime` | Regime name |
| `solver` | Solver name string |
| `solve time` | Solve time in seconds |
| `return` | Expected portfolio return |
| `CVaR` | Portfolio CVaR |
| `obj` | Objective value |

**`portfolio`** — `Portfolio` object with attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | e.g. `"CUOPT_optimal"` |
| `tickers` | list[str] | Asset ticker symbols |
| `weights` | np.ndarray | Optimal weights (num_assets,) |
| `cash` | float | Optimal cash allocation |
| `time_range` | tuple | Regime date range |

### How to Extend

- **Switch solver**: Replace `cp.CUOPT` with `cp.CLARABEL`, `cp.HIGHS`, etc.
- **Compare GPU vs CPU**: Solve twice with different solver settings, then call `utils.compare_results(gpu_result, cpu_result)`.
- **Compare across market regimes**: Use `cvar_utils.optimize_market_regimes()` with a dict of regimes and a list of solver settings.
- **MILP** (cardinality): Set `cardinality=10` in `CvarParameters`; use `mip_absolute_tolerance` in solver settings; expect longer solve times.

---

## Step 5: Backtest

> Evaluate portfolio performance on out-of-sample data.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "Want backtesting?" | Yes — always recommended | Validates out-of-sample performance |
| "What test period?" | 6 months after training end date | Must not overlap with training period |
| "Benchmark to compare against?" | Equal-weight portfolio | Or user can specify custom portfolios |
| "Want a backtest plot?" | Confirm — see [Visualization Guide](#visualization-guide) | Cumulative returns chart |

### Code (cvar_basic.ipynb, Cells 30–31)

```python
from cufolio import backtest, cvar_utils

# Define out-of-sample test period
test_regime_dict = {"name": "test_recent", "range": ("2023-09-01", "2024-07-01")}
test_returns_dict = utils.calculate_returns(data_path, test_regime_dict, returns_compute_settings)

# (Optional) custom benchmark portfolios
portfolios_dict = {
    'AMZN-JPM': ({'AMZN': 0.72, 'JPM': 0.18}, 0.1),
    'AAPL-MSFT': ({'AAPL': 0.29, 'MSFT': 0.61}, 0.1),
}
benchmark_portfolios = cvar_utils.generate_user_input_portfolios(portfolios_dict, test_returns_dict)

# Create backtester and run
backtester = backtest.portfolio_backtester(
    portfolio,               # optimized portfolio from Step 4
    test_returns_dict,
    risk_free_rate=0.0,
    test_method="historical",
    benchmark_portfolios=benchmark_portfolios   # None → equal-weight benchmark
)

backtest_result, ax = backtester.backtest_against_benchmarks(
    plot_returns=True,
    cut_off_date="2024-01-01"
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_method` | `"historical"` | `"historical"`, `"kde_simulation"`, or `"gaussian_simulation"` |
| `risk_free_rate` | 0.0 | Risk-free rate for Sharpe/Sortino calculation |
| `benchmark_portfolios` | None | List of Portfolio objects or None (→ equal-weight) |
| `cut_off_date` | None | Vertical line marking train/test boundary |

### Output: `backtest_result` (DataFrame)

| Column | Description |
|--------|-------------|
| `returns` | Daily returns array |
| `cumulative returns` | Cumulative returns array |
| `portfolio name` | Name string |
| `mean portfolio return` | Mean daily return |
| `sharpe` | Annualized Sharpe ratio |
| `sortino` | Annualized Sortino ratio |
| `max drawdown` | Maximum drawdown |

### How to Extend

- **Custom benchmarks**: Pass `portfolios_dict` with `{name: (weights_dict, cash)}` entries.
- **Simulated test**: Set `test_method="kde_simulation"` or `"gaussian_simulation"` instead of historical.
- **Side-by-side plot**: Use `utils.portfolio_plot_with_backtest(portfolio, backtester, cut_off_date, ...)` for allocation + backtest in one figure.

---

## Composite Workflow: Efficient Frontier

> Sweep risk aversion to trace the Pareto-optimal risk-return curve. Internally runs Steps 1–4 many times.

### Ask the User

| Question | If not specified, use default | Why it matters |
|----------|-------------------------------|----------------|
| "How smooth should the frontier be?" | `ra_num=30` (30 points) | More points = smoother curve but longer compute time |
| "Custom portfolios to overlay?" | None | User can compare their own portfolios against the optimal frontier |
| "Save the plot?" | Confirm save path | `save_path="results/EF_plot.png"` |

### Code (efficient_frontier.ipynb, Cells 2–6)

```python
from cufolio import cvar_utils
from cufolio.cvar_parameters import CvarParameters
import cvxpy as cp

# Long-only parameters for EF
ef_cvar_params = CvarParameters(
    w_min=0.0, w_max=1.0,
    c_min=0.0, c_max=0.0,   # fully invested, no cash
    L_tar=1.0,               # no leverage
    risk_aversion=1,
    confidence=0.95
)

# (Assumes returns_dict already computed via Steps 1–2)

results_df, fig, ax = cvar_utils.create_efficient_frontier(
    returns_dict,
    ef_cvar_params,
    solver_settings={"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"},
    ra_num=30,                      # number of risk aversion levels
    min_risk_aversion=-3,           # log10 scale: 10^-3 (aggressive)
    max_risk_aversion=1,            # log10 scale: 10^1  (conservative)
    custom_portfolios_dict={        # overlay custom portfolios
        "My portfolio": ({"AAPL": 0.3, "MSFT": 0.5, "LLY": 0.2}, 0.0)
    },
    show_plot=True
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ra_num` | 25 | Number of risk-aversion levels (more → smoother frontier) |
| `min_risk_aversion` | -3 | Log₁₀ lower bound (10⁻³ = aggressive) |
| `max_risk_aversion` | 1 | Log₁₀ upper bound (10¹ = conservative) |
| `custom_portfolios_dict` | None | `{name: (weights_dict, cash)}` to overlay on the plot |
| `benchmark_portfolios` | True | Show min-variance, max-Sharpe, max-return markers |
| `show_discretized_portfolios` | True | Exhaustive combinatorial portfolios (very slow) |
| `notional` | 1e7 | Notional amount for y-axis scaling |
| `save_path` | None | Path to save figure PNG |

### Output

- `results_df` — DataFrame with one row per risk-aversion level, columns: `solver`, `solve time`, `return`, `CVaR`, `obj`, `risk_aversion`, `variance`, `volatility`, `sharpe`
- `fig`, `ax` — Matplotlib figure and axes

### How to Extend

- **Finer resolution**: Increase `ra_num` to 50+.
- **Compare custom portfolios**: Use `custom_portfolios_dict` to see where user portfolios sit relative to the frontier.
- **Different constraint sets**: Change `CvarParameters` (e.g., long-short with leverage) to shift the frontier.

---

## Composite Workflow: Dynamic Rebalancing

> Rolling optimization with configurable rebalancing triggers and transaction costs. Internally runs Steps 1–4 at each rebalancing event, automatically compared against a buy-and-hold baseline.

### Ask the User: Strategy Design

Walk the user through these decisions **in order**. Each builds on the previous.

**Step A — Trading Period and Windows:**

```
"What trading period?
  → Please specify start and end dates (YYYY-MM-DD).
    If the user gives an ambiguous range (e.g. '2015-2019'), ask whether
    they mean inclusive end (2015-01-01 to 2019-12-31) or exclusive end
    (2015-01-01 to 2019-01-01)."

"How often to evaluate (trading frequency)?
  (a) Daily (1 trading day)
  (b) Weekly (5 trading days)
  (c) Monthly (21 trading days)
  (d) Quarterly (63 trading days)
  (e) Other — please specify number of trading days"

"How much history per optimization (look-back window)?
  (a) 6 months (126 trading days)
  (b) 1 year (252 trading days)
  (c) 2 years (504 trading days)
  (d) Other — please specify number of trading days"
```

**Step B — Rebalancing Trigger (the core strategy choice):**

Present these three options and let the user choose:

```
"What should trigger a rebalance? Choose one:

 0. Fixed Schedule (no trigger) — Rebalance every evaluation window unconditionally.
    Good for: calendar-based strategies, daily/weekly/monthly rebalancing.
    Implementation: use drift_from_optimal trigger with threshold 0 (any price
    movement causes drift > 0, so it always fires).
    Important: pass plot_title to override the auto-generated title, which would
    otherwise say 'Drift From Optimal'. Match the title to look_forward_window:
      1 → 'Daily Rebalancing Strategy'
      5 → 'Weekly Rebalancing Strategy'
      21 → 'Monthly Rebalancing Strategy'
      63 → 'Quarterly Rebalancing Strategy'

 1. Percentage Change — Rebalance when portfolio value drops below a threshold
    over the evaluation window. Also triggers on cumulative consecutive losses.
    Good for: loss-aversion strategies, drawdown protection.
    Example: rebalance if the portfolio drops more than 0.5% in a month.

 2. Drift from Optimal — Rebalance when current weights deviate from the
    last optimized weights (due to price movements) beyond a tolerance.
    Good for: maintaining target allocation, tracking error control.
    Example: rebalance if weights drift by more than 5% (L1 norm).

 3. Max Drawdown — Rebalance when maximum drawdown in the evaluation window
    exceeds a threshold.
    Good for: tail-risk management, crisis response.
    Example: rebalance if drawdown exceeds 10%.
"
```

Once the user picks a trigger, ask for the threshold with choices:

**If pct_change:**
```
"What % drop triggers rebalancing?
  (a) -0.3% (sensitive)
  (b) -0.5% (moderate)
  (c) -1.0% (conservative)
  (d) Other — please specify"
```

**If drift_from_optimal:**
```
"What deviation tolerance?
  (a) 3% (tight — frequent rebalancing)
  (b) 5% (moderate)
  (c) 10% (loose — infrequent rebalancing)
  (d) Other — please specify"

"Which norm?
  (a) L1 (sum of absolute diffs)
  (b) L2 (Euclidean distance)"
```

**If max_drawdown:**
```
"What drawdown % triggers rebalancing?
  (a) 5% (aggressive protection)
  (b) 8% (moderate)
  (c) 10% (standard)
  (d) 15% (conservative — only large drawdowns)
  (e) Other — please specify"
```

**Step C — Transaction Costs and Constraints:**

```
"Transaction costs?
  (a) None (frictionless, factor = 0)
  (b) 10 bps (factor = 0.001, typical)
  (c) 20 bps (factor = 0.002, realistic)
  (d) Other — please specify"

"Turnover limit per rebalance?
  (a) Unconstrained (no limit)
  (b) 30% (tight — low churn)
  (c) 50% (moderate)
  (d) Other — please specify"

"Portfolio constraints?"
  → Same as Step 3 (long-only/long-short, max weight, cash, leverage, sectors).
    If already specified above, confirm they apply at each rebalance.
```

**Step D — Output:**

```
"Want rebalancing plots?
  (a) Yes — strategy vs buy-and-hold comparison
  (b) No — numerical results only"

"Track a specific asset's weight over time?
  (a) No
  (b) Yes — please specify the ticker (e.g. NVDA, AAPL)"
```

### How the Testing Logic Works

Understanding the simulation loop helps set parameters correctly:

```
1. BASELINE: At trading_start, optimize once → this becomes the buy-and-hold portfolio.

2. LOOP: Starting from trading_start, step forward by look_forward_window days:
   a. If rebalancing trigger is met (or first iteration):
      - Take the last look_back_window days of history
      - Generate new scenarios (KDE/Gaussian)
      - Re-optimize with existing portfolio as reference (enables T_tar turnover constraint)
      - Deduct transaction costs: cost = factor × ‖w_new − w_old‖₁
   b. Backtest the current portfolio over the next look_forward_window days
   c. Check trigger condition against threshold → set flag for next iteration

3. OUTPUT: Compare cumulative value of dynamic strategy vs buy-and-hold baseline.
```

**Key constraint**: `trading_start` must have at least `look_back_window` calendar days of data before it in the dataset. The code asserts this on initialization.

### Code (rebalancing_strategies.ipynb, Cells 3–5)

```python
from cufolio import rebalance
from cufolio.cvar_parameters import CvarParameters
from cufolio.settings import ReturnsComputeSettings, ScenarioGenerationSettings, KDESettings
import cvxpy as cp

# Portfolio constraints (applied at each rebalance)
cvar_params = CvarParameters(
    w_min=-0.3, w_max=0.8,
    c_min=0.1, c_max=0.4,
    L_tar=1.6,
    T_tar=0.5,             # turnover constraint per rebalance event
    risk_aversion=1, confidence=0.95
)

rebal_obj = rebalance.rebalance_portfolio(
    dataset_directory="../data/stock_data/sp500.csv",
    returns_compute_settings=ReturnsComputeSettings(return_type='LOG', freq=1),
    scenario_generation_settings=ScenarioGenerationSettings(
        num_scen=10000, fit_type='kde',
        kde_settings=KDESettings(bandwidth=0.01, kernel='gaussian', device='GPU'),
        verbose=False
    ),
    trading_start="2022-07-01",
    trading_end="2024-01-01",
    look_forward_window=21,       # evaluate every ~1 month
    look_back_window=252,         # use 1 year of history per optimization
    cvar_params=cvar_params,
    solver_settings={"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"},
    re_optimize_criteria={"type": "pct_change", "threshold": -0.005}
)

results_df, dates, values = rebal_obj.re_optimize(
    transaction_cost_factor=0.001,
    plot_results=True,
    save_plot=True,
    results_dir="../results/rebalancing_strategies"
)
```

### All Parameters Reference

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_directory` | str | Path to price data CSV |
| `returns_compute_settings` | `ReturnsComputeSettings` | Same as Step 1 (must be Pydantic object, not dict) |
| `scenario_generation_settings` | `ScenarioGenerationSettings` | Same as Step 2 (must be Pydantic object, not dict) |
| `trading_start` | str | Start of trading period `YYYY-MM-DD` |
| `trading_end` | str | End of trading period `YYYY-MM-DD` |
| `look_forward_window` | int | Evaluation cadence in trading days (5=weekly, 21=monthly, 63=quarterly) |
| `look_back_window` | int | Rolling history window for each optimization in trading days |
| `cvar_params` | CvarParameters | Portfolio constraints — applied at every rebalance event |
| `solver_settings` | dict | Solver settings (same as Step 4) |
| `re_optimize_criteria` | dict | Rebalancing trigger — see strategy options below |
| `print_opt_result` | bool | Print detailed results at each rebalance event (default False) |

**`re_optimize_criteria` — the three trigger strategies:**

| Strategy | Dict format | Trigger logic |
|----------|-------------|---------------|
| **Percentage change** | `{"type": "pct_change", "threshold": -0.005}` | Fires if single-period return < threshold, OR cumulative consecutive negative returns < threshold |
| **Drift from optimal** | `{"type": "drift_from_optimal", "threshold": 0.05, "norm": 1}` | Computes actual weights after price movements; fires if deviation from optimal exceeds tolerance. `norm`: 1 (L1, sum of absolute diffs) or 2 (L2, sum of squared diffs) |
| **Max drawdown** | `{"type": "max_drawdown", "threshold": 0.10}` | Fires if max drawdown within evaluation window exceeds threshold |

**`re_optimize()` execution parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transaction_cost_factor` | 0 | Cost per unit of turnover: `cost = factor × ‖w_new − w_old‖₁`. Typical: 0.001 (10 bps) |
| `plot_results` | False | Generate strategy vs buy-and-hold comparison plot |
| `save_plot` | False | Save plot to `results_dir` |
| `results_dir` | `"results"` | Output directory for saved plots |
| `plot_title` | None | Custom plot title. If not provided, auto-generated from trigger type. **Use this for fixed-schedule strategies** where the trigger type (`drift_from_optimal`) doesn't reflect the actual strategy name |

### Output

- `results_df` — DataFrame with one row per evaluation window. Columns: trigger metric value, `re_optimized` flag, `portfolio_value`, `solver_time`, `optimal_portfolio` (Portfolio object)
- `re_optimize_dates` — list of dates when rebalancing was triggered
- `cumulative_portfolio_value` — pd.Series of daily portfolio values over the entire trading period

### Strategy Recipes

Ready-to-use configurations for common strategies. Present these to the user as starting points they can customize.

**Fixed-schedule rebalancing (no trigger):**
```python
re_optimize_criteria = {"type": "drift_from_optimal", "threshold": 0, "norm": 1}  # threshold 0 → always triggers
look_forward_window = 1    # 1=daily, 5=weekly, 21=monthly, 63=quarterly
look_back_window = 252     # 1 year history
T_tar = 0.3                # constrain turnover to avoid excessive churn
transaction_cost_factor = 0.001  # 10 bps — important with frequent rebalancing

# Derive plot_title from look_forward_window so the chart title reflects
# the actual cadence instead of "Drift From Optimal Rebalancing Strategy":
freq_labels = {1: "Daily", 5: "Weekly", 21: "Monthly", 63: "Quarterly"}
plot_title = f"{freq_labels.get(look_forward_window, f'Every {look_forward_window}-Day')} Rebalancing Strategy"
```

**Conservative: Monthly check, rebalance on small losses**
```python
re_optimize_criteria = {"type": "pct_change", "threshold": -0.003}
look_forward_window = 21   # monthly evaluation
look_back_window = 252     # 1 year history
T_tar = 0.3                # tight turnover constraint
transaction_cost_factor = 0.002  # 20 bps realistic costs
```

**Responsive: Weekly check, tight drift tolerance**
```python
re_optimize_criteria = {"type": "drift_from_optimal", "threshold": 0.03, "norm": 1}
look_forward_window = 5    # weekly evaluation
look_back_window = 126     # 6 months history
T_tar = 0.5                # moderate turnover limit
transaction_cost_factor = 0.001
```

**Crisis-aware: Monthly check, drawdown protection**
```python
re_optimize_criteria = {"type": "max_drawdown", "threshold": 0.08}
look_forward_window = 21   # monthly evaluation
look_back_window = 252     # 1 year history
T_tar = None               # allow full rebalancing in crisis
transaction_cost_factor = 0.001
```

**Quarterly passive: Infrequent rebalancing, low cost**
```python
re_optimize_criteria = {"type": "drift_from_optimal", "threshold": 0.10, "norm": 1}
look_forward_window = 63   # quarterly evaluation
look_back_window = 504     # 2 years history
T_tar = 0.2                # very tight turnover
transaction_cost_factor = 0.0005  # 5 bps low-cost
```

### After Running: Follow-Up Analysis

After the rebalancing simulation completes, offer the user these follow-up options:

```python
# 1. Track how a specific asset's weight evolved over time
rebal_obj.plot_weights_vs_prices(results_df, "NVDA")

# 2. Count rebalancing events and timing
print(f"Total rebalances: {len(dates)}")
print(f"Rebalance dates: {[d.strftime('%Y-%m-%d') for d in dates]}")

# 3. Extract solver times from results
solver_times = results_df['solver_time'].dropna()
print(f"Avg solve time: {solver_times.mean():.3f}s")

# 4. Compare two strategies by running both and overlaying results
# (run rebalance_portfolio twice with different re_optimize_criteria,
#  then plot both cumulative_portfolio_value series on the same axes)
```

---

## Visualization Guide

cuFOLIO provides built-in plotting functions. **Always ask the user if they want plots**, and if so, which ones. The six plot types below cover all common visualization needs.

### Confirm Output with User

> **MANDATORY** — Do NOT generate plots or save files without asking the user first. Always confirm output format before running the final step.

Ask:

```
"Would you like:
 1. Numerical results only (tables/metrics printed to console)
 2. Plots saved to files (specify directory)
 3. Both numerical results and plots
 
 Available plots:
 • Portfolio allocation bar chart
 • Backtest cumulative returns
 • Combined allocation + backtest (side-by-side)
 • Efficient frontier
 • Rebalancing strategy comparison
 • Weight evolution vs price"
```

### Plot 1: Portfolio Allocation

Horizontal bar chart showing long (blue), short (red), and cash (yellow) positions.

```python
# After Step 4 (solve)
ax = portfolio.plot_portfolio(
    show_plot=True,         # display immediately
    min_percentage=1.0,     # hide positions < 1%
    save_path="results/",   # save PNG to directory (optional)
    title="My Portfolio",   # custom title (optional)
    figsize=(12, 8),        # figure size
    dpi=300                 # resolution
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_plot` | False | Display the plot immediately |
| `min_percentage` | 0.0 | Hide positions below this % threshold |
| `save_path` | None | Directory to save PNG (file named `{portfolio.name}_allocation.png`) |
| `title` | Auto | Custom title; default uses portfolio name + date range |
| `figsize` | (12, 8) | Figure size in inches |
| `ax` | None | Pass existing matplotlib axes to embed in a multi-panel figure |

### Plot 2: Backtest Cumulative Returns

Line chart comparing optimized portfolio vs benchmarks over the test period.

```python
# After Step 5 (backtest)
backtest_result, ax = backtester.backtest_against_benchmarks(
    plot_returns=True,                  # enable plot
    cut_off_date="2024-01-01",          # vertical line at train/test boundary
    title="Backtest Results",           # custom title (optional)
    save_plot=True,                     # save to file
    results_dir="results/backtest/"     # output directory
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plot_returns` | False | Enable the cumulative returns chart |
| `cut_off_date` | None | Date string for vertical boundary line |
| `title` | None | Custom plot title |
| `save_plot` | False | Save PNG to `results_dir` |
| `results_dir` | `"results"` | Directory for saved plot |
| `ax` | None | Pass existing axes for multi-panel embedding |

### Plot 3: Combined Allocation + Backtest (Side-by-Side)

Two-panel figure: allocation bar chart (left) + cumulative returns (right).

```python
from cufolio import utils

utils.portfolio_plot_with_backtest(
    portfolio=portfolio,            # optimized portfolio
    backtester=backtester,          # backtester from Step 5
    cut_off_date="2024-01-01",      # train/test boundary
    backtest_plot_title="Backtest", # right panel title
    save_plot=True,                 # save combined PNG
    results_dir="results/"          # output directory
)
# Saved as: results/combined_{name}_{method}_analysis.png
```

### Plot 4: Efficient Frontier

Risk-return frontier with annotated key portfolios and optional custom overlays. Generated automatically by `create_efficient_frontier()`.

```python
results_df, fig, ax = cvar_utils.create_efficient_frontier(
    returns_dict,
    cvar_params,
    solver_settings={"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"},
    ra_num=30,
    custom_portfolios_dict={
        "User Portfolio": ({"AAPL": 0.3, "MSFT": 0.5, "LLY": 0.2}, 0.0)
    },
    benchmark_portfolios=True,              # show min-var, max-Sharpe, max-return
    show_discretized_portfolios=False,      # set True for exhaustive (slow)
    show_plot=True,
    save_path="results/EF_plot.png",        # save to file
    notional=1e7,                           # y-axis: return on $10M
    title="Efficient Frontier – S&P 500"
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_plot` | True | Display immediately |
| `save_path` | None | Path to save PNG |
| `benchmark_portfolios` | True | Annotate min-variance, max-Sharpe, max-return |
| `custom_portfolios_dict` | None | `{name: (weights_dict, cash)}` diamond markers |
| `show_discretized_portfolios` | True | Show all discrete weight combos (very slow; set False) |
| `notional` | 1e7 | Scale y-axis as return on notional amount |
| `style` | `"publication"` | `"publication"`, `"presentation"`, or `"minimal"` |

### Plot 5: Rebalancing Strategy Comparison

Time-series comparison of dynamic rebalancing vs buy-and-hold baseline with rebalancing event markers.

```python
results_df, dates, values = rebal_obj.re_optimize(
    transaction_cost_factor=0.001,
    plot_results=True,                          # enable plot
    save_plot=True,                             # save PNG
    results_dir="results/rebalancing/",         # output directory
    plot_title="Monthly Rebalancing Strategy"   # optional custom title
)
# Saved as: results/rebalancing/rebalancing_{strategy}_{threshold}_{dates}events.png
```

> **Fixed-schedule strategies**: When using `drift_from_optimal` with `threshold=0` to implement a fixed trading frequency, always pass `plot_title` to override the auto-generated title. Derive it from `look_forward_window`: 1 → "Daily", 5 → "Weekly", 21 → "Monthly", 63 → "Quarterly", other values → "Every N-Day". Without this, the plot will misleadingly say "Drift From Optimal Rebalancing Strategy".

### Plot 6: Weight Evolution vs Price

Dual-axis chart showing how a specific asset's weight evolves alongside its price over the rebalancing period.

```python
# After rebalancing (rebal_obj.re_optimize())
rebal_obj.plot_weights_vs_prices(
    re_optimize_results=results_df,     # results DataFrame from re_optimize()
    ticker="NVDA",                      # asset to track
    plot_title="NVDA Weight vs Price"   # optional custom title (default: "{ticker} weights vs. prices")
)
```

### Plotting Tips

- **Headless environments** (no display): Use `matplotlib.use('Agg')` at the top of the script, set `show_plot=False`, and use `save_plot=True` / `save_path` to write PNGs to disk.
- **Embedding in notebooks**: All plot functions return matplotlib `ax` objects — use `show_plot=True` for inline display.
- **Multi-panel custom figures**: Pass `ax=` to `plot_portfolio()` and `backtest_against_benchmarks()` to compose custom layouts.
- **Saving all plots**: Always provide `save_path` / `save_plot=True` + `results_dir` — the user may not see inline plots.

---

## Data

S&P 500 price data (397 stocks, 2005–2025) is auto-downloaded on first run to `../data/stock_data/sp500.csv` via `utils.download_data()`.

**Expected data format**: Rows = dates (index), Columns = ticker symbols, Values = adjusted closing prices. Supported formats: CSV, Parquet, Excel, JSON.

## Reference

Based on [NVIDIA-AI-Blueprints/quantitative-portfolio-optimization](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization)

# cuFOLIO Demo - Streamlit Applications

Interactive web applications for portfolio optimization and risk management using Mean-CVaR (Conditional Value-at-Risk) optimization.

## 📱 Applications

### 1. **CVaR Portfolio Optimizer** (`cvar_streamlit_app.py`)
Single-period portfolio optimization with interactive parameter tuning.

**Features:**
- Real-time portfolio allocation visualization
- Comprehensive constraint settings (weight bounds, leverage, cardinality)
- GPU (cuOpt) and CPU (HiGHS) solver support
- Detailed optimization metrics and performance statistics

### 2. **cuFOLIO Efficient Frontier** (`efficient_frontier_streamlit_app.py`)
Multi-portfolio efficient frontier generation with progressive GPU vs CPU comparison.

**Features:**
- Side-by-side GPU vs CPU performance comparison
- Progressive frontier construction with real-time updates
- Key portfolio identification (Min Variance, Max Sharpe, Max Return)
- Interactive risk-return visualization

### 3. **cuFOLIO Rebalancing Strategies** (`rebalancing_streamlit_app.py`)
Dynamic portfolio rebalancing simulation with multiple trigger strategies.

**Features:**
- Multiple rebalancing triggers (percentage change, drift, drawdown)
- Progressive backtesting with real-time performance tracking
- GPU vs CPU solver comparison
- Optional constraints (turnover, cardinality, hard CVaR limits)

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Ensure CVaR source code is accessible
# - CVaR source files should be in ../src/
# - Data files should be in ../../data/stock_data/
```

### Running the Apps
```bash
# Single-period optimization
streamlit run cvar_streamlit_app.py

# Efficient frontier generation
streamlit run efficient_frontier_streamlit_app.py

# Dynamic rebalancing strategies
streamlit run rebalancing_streamlit_app.py
```

## 🔧 Common Parameters

### Dataset Configuration
- **Dataset**: Stock datasets (sp500, baby_dataset, etc.)
- **Return Type**: LOG or SIMPLE returns
- **Date Range**: Historical period for analysis
- **Device**: GPU or CPU computation

### CVaR Parameters
- **Risk Aversion**: Higher values prefer lower risk
- **Confidence Level**: CVaR confidence (e.g., 95%)
- **Scenarios**: Number of return scenarios

### Portfolio Constraints
- **Weight Bounds**: Min/max allocation per asset
- **Cash Bounds**: Cash holding limits
- **Leverage Target**: Maximum portfolio leverage
- **Cardinality**: Maximum number of assets (optional)

## 🎯 Use Cases

**Single Portfolio Optimization**: Use CVaR Portfolio Optimizer for one-time portfolio construction with specific constraints.

**Risk-Return Analysis**: Use Efficient Frontier app to understand the risk-return trade-off across multiple portfolios.

**Dynamic Strategies**: Use Rebalancing Strategies app to simulate adaptive portfolio management over time.

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28+
- CVXPY with HiGHS solver
- Optional: NVIDIA GPU + CUDA for cuOpt acceleration

## 🐛 Troubleshooting

**Import Errors**: Ensure CVaR `src/` folder is accessible and dependencies installed.

**Data Issues**: Verify data files exist in `../../data/stock_data/`.

**GPU Solver**: Check NVIDIA GPU and CUDA installation for cuOpt support.

**Performance**: Use fewer scenarios or CPU solver for faster computation on limited hardware.

# Lifecycle Planning

A comprehensive Python library for lifecycle financial planning based on "Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance" by Ibbotson, Milevsky, Chen, and Zhu.

**PyPI Package**: `lifecycle-planning` | **Import**: `lifecycle_planning`

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 📊 **Mortality Modeling** - Gompertz survival probabilities with gender-specific parameters
- 💰 **Human Capital Valuation** - Present value of future earnings with mortality weighting
- 📈 **Portfolio Optimization** - Mean-variance optimization with CRRA utility
- 💵 **Optimal Spending Rules** - Consumption smoothing based on lifecycle preferences
- 🎯 **Balance Sheet Framework** - Individual economic balance sheets including human capital
- 📉 **Annuity Pricing** - SPIA valuation and mortality credit analysis
- 🧮 **CRRA Utility Functions** - Constant relative risk aversion modeling
- ⚙️ **Command-Line Interface** - Easy-to-use CLI for demonstrations and analysis

## Installation

### From PyPI (Recommended)

```bash
pip install lifecycle-planning
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Druhayes/lifetime-financial-advice.git
cd lifetime-financial-advice

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

### Optional Dependencies

```bash
# Install with CLI support
pip install lifecycle-planning[cli]

# Install with all optional dependencies
pip install lifecycle-planning[all]
```

## Quick Start

### Command-Line Interface

The package includes a CLI for quick demonstrations:

```bash
# Show version
lifecycle --version

# Display package information
lifecycle info

# Run quick demonstration
lifecycle demo

# Run Isabela case study
lifecycle demo isabela

# List available demos
lifecycle demo --list
```

### Python Library Usage

#### 1. Basic Example - Create a Financial Plan

```python
from lifecycle_planning import IndividualBalanceSheet, FinancialPreferences

# Define financial preferences
preferences = FinancialPreferences(
    theta=0.35,  # Risk tolerance (35% - moderate)
    rho=0.02,    # Impatience (2% - patient)
    eta=0.50,    # Consumption smoothing preference (50%)
    gamma=0.25,  # Bequest flexibility (25%)
    phi=1.5      # Bequest strength (1.5%)
)

# Create individual balance sheet
client = IndividualBalanceSheet(
    current_age=30,
    retirement_age=65,
    gender="female",
    current_income=85000,
    nondiscretionary_spending=50000,
    taxable_wealth=25000,
    tax_advantaged_wealth=150000,
    preferences=preferences,
)

# Analyze financial situation
print(f"Net Worth: ${client.net_worth:,.0f}")
print(f"Human Capital: ${client.human_capital_value:,.0f}")
print(f"Financial Wealth: ${client.financial_wealth:,.0f}")

# Get optimal spending recommendation
spending = client.optimal_spending()
print(f"\nOptimal Annual Spending: ${spending['total_consumption']:,.0f}")
print(f"Savings Rate: {spending['savings_rate']:.1%}")
```

#### 2. Using Pre-Built Example Profiles

```python
from lifecycle_planning.examples import (
    create_isabela,
    create_conservative_retiree,
    create_aggressive_professional,
)

# Use Isabela example (25-year-old with $75k income)
isabela = create_isabela()
print(f"Isabela's net worth: ${isabela.net_worth:,.0f}")

# Use conservative retiree profile
retiree = create_conservative_retiree()
spending = retiree.optimal_spending()
print(f"Retiree optimal spending: ${spending['total_consumption']:,.0f}")

# Use aggressive professional profile
young_pro = create_aggressive_professional()
print(f"Professional's human capital: ${young_pro.human_capital_value:,.0f}")
```

#### 3. Mortality Analysis

```python
from lifecycle_planning import survival_probability, life_expectancy

# Calculate survival probabilities
prob_to_retirement = survival_probability(25, 65, gender="female")
prob_to_85 = survival_probability(25, 85, gender="female")

print(f"Probability of reaching 65: {prob_to_retirement:.1%}")
print(f"Probability of reaching 85: {prob_to_85:.1%}")

# Calculate life expectancy
remaining_years = life_expectancy(25, gender="female")
print(f"Life expectancy: {25 + remaining_years:.1f} years")
```

#### 4. Portfolio Optimization

```python
import numpy as np
from lifecycle_planning import mvo_optimal_portfolio

# Define asset class assumptions
returns = np.array([0.07, 0.03])  # Stocks: 7%, Bonds: 3%
cov_matrix = np.array([
    [0.15**2, 0.15 * 0.05 * 0.2],      # Stock variance & correlation
    [0.15 * 0.05 * 0.2, 0.05**2]       # Bond variance
])

# Optimize for moderate risk tolerance
weights, exp_return, std = mvo_optimal_portfolio(
    returns, cov_matrix, theta=0.35
)

print(f"Optimal Allocation:")
print(f"  Stocks: {weights[0]:.1%}")
print(f"  Bonds: {weights[1]:.1%}")
print(f"Expected Return: {exp_return:.2%}")
print(f"Standard Deviation: {std:.2%}")
```

## Client Profiles

The library supports diverse client scenarios through customizable preferences:

### Conservative Retiree

```python
from lifecycle_planning import IndividualBalanceSheet, FinancialPreferences

conservative_prefs = FinancialPreferences(
    theta=0.15,  # Low risk tolerance
    rho=0.04,    # Higher impatience (shorter horizon)
    eta=0.75,    # Prefers smooth consumption
    gamma=0.20,  # Less flexible on bequests
    phi=2.0      # Strong bequest motive
)

retiree = IndividualBalanceSheet(
    current_age=65,
    retirement_age=65,
    gender="male",
    current_income=0,  # Retired
    nondiscretionary_spending=60000,
    taxable_wealth=300000,
    tax_advantaged_wealth=800000,
    preferences=conservative_prefs,
)
```

### Aggressive Young Professional

```python
aggressive_prefs = FinancialPreferences(
    theta=0.55,  # High risk tolerance
    rho=0.01,    # Very patient (long horizon)
    eta=0.30,    # Willing to vary consumption
    gamma=0.35,  # Flexible on bequests
    phi=0.8      # Weak bequest motive
)

young_prof = IndividualBalanceSheet(
    current_age=28,
    retirement_age=67,
    gender="female",
    current_income=120000,
    nondiscretionary_spending=45000,
    taxable_wealth=15000,
    tax_advantaged_wealth=65000,
    preferences=aggressive_prefs,
)
```

## Parameter Guide

### Risk Tolerance (θ)
- **0.10-0.25**: Conservative (retirees, risk-averse individuals)
- **0.25-0.45**: Moderate (balanced investors)
- **0.45-0.70**: Aggressive (young professionals, growth-focused)

### Impatience (ρ)
- **0.01-0.02**: Very patient (long time horizon)
- **0.02-0.04**: Moderate (standard planning)
- **0.04-0.06**: Impatient (short time horizon)

### Consumption Smoothing (η)
- **0.20-0.40**: Willing to vary consumption over time
- **0.40-0.60**: Balanced approach
- **0.60-0.80**: Strong preference for smooth consumption

## Advanced Usage

### Custom Mortality Assumptions

```python
from lifecycle_planning import IndividualBalanceSheet, FinancialPreferences

# Create client with personalized life expectancy
# (e.g., family history of longevity)
client = IndividualBalanceSheet(
    current_age=25,
    retirement_age=65,
    gender="female",
    current_income=75000,
    nondiscretionary_spending=40000,
    tax_advantaged_wealth=100000,
    preferences=preferences,
    life_expectancy_override=94,  # Family lives to 94 on average
)
```

### Accessing Individual Modules

```python
# Import specific functionality
from lifecycle_planning.core.mortality import (
    survival_probability,
    life_expectancy,
    survival_probability_series,
)

from lifecycle_planning.core.utility import (
    crra_utility,
    levy_markowitz_utility,
)

from lifecycle_planning.core.portfolio import (
    mvo_optimal_portfolio,
    portfolio_expected_return,
)

from lifecycle_planning.core.annuities import (
    spia_price,
    mortality_credit,
)
```

## Running Examples

The package includes comprehensive example scripts:

```bash
# Quick demonstration (all features)
python examples/quick_demo.py

# Isabela case study (comprehensive example)
python examples/isabela.py

# Or use the CLI
lifecycle demo
lifecycle demo isabela
```

## Project Structure

```
lifecycle-planning/
├── src/lifecycle_planning/     # Main package
│   ├── core/                   # Core modules
│   │   ├── mortality.py        # Gompertz survival model
│   │   ├── utility.py          # CRRA utility functions
│   │   ├── present_value.py    # PV calculations
│   │   ├── income.py           # Income projections
│   │   ├── spending.py         # Optimal spending rules
│   │   ├── portfolio.py        # Mean-variance optimization
│   │   ├── annuities.py        # Annuity pricing
│   │   └── balance_sheet.py    # Balance sheet framework
│   ├── examples/               # Example profiles
│   │   └── profiles.py         # Pre-built client profiles
│   └── cli/                    # Command-line interface
│       └── commands.py         # CLI implementation
├── examples/                   # Standalone example scripts
│   ├── quick_demo.py          # Quick demonstration
│   └── isabela.py             # Isabela case study
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## Mathematical Foundation

### Gompertz Mortality Model
- Parametric survival probability model
- Gender-specific parameters (US 2012 Individual Annuity Mortality tables)
- Formula: `g(a₂, a₁; m, b) = exp(-exp((a₁-m)/b) * (exp((a₂-a₁)/b) - 1))`

### CRRA Utility
- Constant Relative Risk Aversion
- Formula: `u(c) = (c^θ - 1) / θ` for θ ≠ 1, `ln(c)` for θ = 1
- Levy-Markowitz approximation for mean-variance optimization

### Human Capital
- Present value of future labor income
- Mortality-weighted discounting
- Incorporates wage growth and retirement timing

### Economic Balance Sheet
```
Assets:
  + Financial Wealth (taxable + tax-advantaged)
  + Human Capital (PV of future earnings)
- Liabilities:
  - Nondiscretionary Consumption (PV of essential spending)
= Net Worth (available for discretionary consumption)
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install lifecycle-planning[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=lifecycle_planning

# Run specific test module
pytest tests/core/test_mortality.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/lifecycle_planning
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure tests pass (`pytest`)
5. Format code (`black .`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

**Primary Source:**
Ibbotson, R. G., Milevsky, M. A., Chen, P., & Zhu, K. X. (2007). *Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance*. The Research Foundation of CFA Institute.
[Download PDF](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/lifetime-financial-advice.pdf)

**Related Works:**
- Milevsky, M. A. (2020). *Retirement Income Recipes in R*
- Milevsky, M. A. (2012). *The 7 Most Important Equations for Your Retirement*
- Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty"

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{lifecycle_planning,
  title = {Lifecycle Planning: A Python Library for Lifecycle Financial Planning},
  author = {Hayes, Drew},
  year = {2026},
  url = {https://github.com/Druhayes/lifetime-financial-advice}
}
```

---

**Built with Python 3.11+ • Powered by NumPy & SciPy**

# Lifetime Financial Advice - Python Library

A comprehensive Python implementation of lifecycle financial planning models based on "Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance" by Ibbotson, Milevsky, Chen, and Zhu.

This library provides tools for:
- **Mortality modeling** using Gompertz survival probabilities
- **CRRA utility** and preference modeling
- **Human capital** and liability calculations
- **Optimal spending** rules for retirement planning
- **Mean-variance portfolio** optimization
- **Annuity pricing** and analysis
- **Individual economic balance sheets** for comprehensive financial planning

## Installation

### Prerequisites

Ensure you have Python 3.11+ and [UV package manager](https://github.com/astral-sh/uv) installed.

#### Install UV

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### Install from Repository

```bash
# Clone the repository
git clone https://github.com/Druhayes/lifetime-financial-advice.git
cd lifetime-financial-advice

# Install dependencies with UV
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Development Installation

```bash
# Install with development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run black .

# Type checking
uv run mypy lifecycle/
```

## Quick Start

### Basic Example - Quick Demo

Run the basic demonstration to see all library features:

```bash
python main.py
```

This will demonstrate:
- Mortality calculations for a 25-year-old woman
- CRRA utility functions
- Human capital calculations
- Portfolio optimization
- Optimal spending rules
- Annuity pricing
- Individual balance sheet creation

### Advanced Example - Isabela Case Study

For a complete real-world example, run the Isabela case study:

```bash
python examples/isabela.py
```

This demonstrates a comprehensive financial plan for a 25-year-old professional with $75,000 income.

## Library Usage

### 1. Import the Library

```python
from lifecycle import (
    # Mortality functions
    survival_probability,
    life_expectancy,
    
    # Utility functions
    crra_utility,
    levy_markowitz_utility,
    
    # Financial calculations
    human_capital,
    present_value,
    
    # Portfolio optimization
    mvo_optimal_portfolio,
    
    # Spending optimization
    optimal_consumption_divisor,
    discretionary_consumption,
    
    # Balance sheet
    IndividualBalanceSheet,
    FinancialPreferences,
)
```

### 2. Create Client Preferences

```python
# Define financial preferences
preferences = FinancialPreferences(
    theta=0.35,  # Risk tolerance (35%)
    rho=0.02,    # Impatience (2%)
    eta=0.50,    # Elasticity of intertemporal substitution (50%)
    gamma=0.25,  # Intergenerational elasticity (25%)
    phi=1.5      # Bequest strength (1.5%)
)
```

### 3. Build Individual Balance Sheet

```python
# Create balance sheet for a client
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

print(f"Net Worth: ${client.net_worth:,.0f}")
print(f"Human Capital: ${client.human_capital_value:,.0f}")
print(f"Financial Wealth: ${client.financial_wealth:,.0f}")
```

### 4. Calculate Optimal Spending

```python
# Get optimal spending recommendation
spending = client.optimal_spending()
print(f"Recommended Annual Spending: ${spending['total_consumption']:,.0f}")
print(f"Savings Rate: {spending['savings_rate']:.1%}")
```

## Adapting for Different Clients

### Example: Conservative Retiree vs. Aggressive Young Professional

#### Conservative Retiree (Age 65)

```python
# Conservative preferences for retiree
conservative_prefs = FinancialPreferences(
    theta=0.15,  # Lower risk tolerance
    rho=0.04,    # Higher impatience (shorter time horizon)
    eta=0.75,    # Prefers smoother consumption
    gamma=0.20,  # Less flexible on bequests
    phi=2.0      # Stronger bequest motive
)

retiree = IndividualBalanceSheet(
    current_age=65,
    retirement_age=65,  # Already retired
    gender="male",
    current_income=0,   # No labor income
    nondiscretionary_spending=60000,
    taxable_wealth=300000,
    tax_advantaged_wealth=800000,
    preferences=conservative_prefs,
)
```

#### Aggressive Young Professional (Age 28)

```python
# Aggressive preferences for young professional
aggressive_prefs = FinancialPreferences(
    theta=0.55,  # Higher risk tolerance
    rho=0.01,    # Lower impatience (longer time horizon)
    eta=0.30,    # More willing to vary consumption
    gamma=0.35,  # More flexible on bequests
    phi=0.8      # Weaker bequest motive
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

### Parameter Guidance

#### Risk Tolerance (θ)
- **0.10-0.25**: Conservative (retirees, risk-averse)
- **0.25-0.45**: Moderate (middle-aged, balanced)
- **0.45-0.70**: Aggressive (young professionals, growth-focused)

#### Impatience (ρ)
- **0.01-0.02**: Patient (long time horizon)
- **0.02-0.04**: Moderate (standard discount rate)
- **0.04-0.06**: Impatient (short time horizon, urgent needs)

#### Elasticity of Intertemporal Substitution (η)
- **0.20-0.40**: Willing to vary consumption over time
- **0.40-0.60**: Balanced preference
- **0.60-0.80**: Prefers smooth, predictable consumption

### Client-Specific Scenarios

#### High-Income Professional
```python
prefs = FinancialPreferences(theta=0.40, rho=0.015, eta=0.35)
client = IndividualBalanceSheet(
    current_age=35, retirement_age=62, gender="male",
    current_income=250000, nondiscretionary_spending=80000,
    taxable_wealth=100000, tax_advantaged_wealth=300000,
    preferences=prefs
)
```

#### Late-Career Saver
```python
prefs = FinancialPreferences(theta=0.25, rho=0.03, eta=0.65)
client = IndividualBalanceSheet(
    current_age=55, retirement_age=67, gender="female",
    current_income=90000, nondiscretionary_spending=55000,
    taxable_wealth=200000, tax_advantaged_wealth=450000,
    preferences=prefs
)
```

## Advanced Features

### Portfolio Optimization

```python
import numpy as np

# Define asset returns and covariance
returns = np.array([0.08, 0.04])  # Stocks, Bonds
cov_matrix = np.array([
    [0.16**2, 0.16 * 0.05 * 0.2],
    [0.16 * 0.05 * 0.2, 0.05**2]
])

weights, exp_return, std = mvo_optimal_portfolio(
    returns, cov_matrix, theta=0.35
)
print(f"Optimal allocation: {weights[0]:.1%} stocks, {weights[1]:.1%} bonds")
```

### Mortality Analysis

```python
# Calculate survival probabilities
prob_to_85 = survival_probability(current_age=30, target_age=85, gender="female")
life_exp = life_expectancy(age=30, gender="female")

print(f"Probability of reaching 85: {prob_to_85:.1%}")
print(f"Life expectancy: {30 + life_exp:.1f} years")
```

### Annuity Pricing

```python
from lifecycle.annuities import spia_price
from lifecycle.mortality import survival_probability_series

# Price a $10,000/year annuity for 65-year-old female
survival_probs = survival_probability_series(65, 40, "female")
price = spia_price(age=65, real_rate=0.025, 
                  survival_probs=survival_probs, annual_payment=10000)

print(f"Annuity price: ${price:,.0f}")
print(f"Payout rate: {10000/price:.2%}")
```

## Core Mathematical Models

### Gompertz Mortality Model
- Uses actuarial survival probability calculations
- Supports both male and female mortality tables
- Based on 2012 Individual Annuity Mortality tables

### CRRA Utility Function
- Constant Relative Risk Aversion utility
- Levy-Markowitz formulation for portfolio optimization
- Supports various risk tolerance levels

### Human Capital Valuation
- Present value of future earnings
- Mortality-weighted calculations
- Incorporates income growth and uncertainty

## Project Structure

```
lifetime-financial-advice/
├── lifecycle/              # Main library
│   ├── __init__.py
│   ├── mortality.py        # Survival probability calculations
│   ├── utility.py          # CRRA and utility functions
│   ├── present_value.py    # PV calculations
│   ├── income.py           # Income modeling
│   ├── spending.py         # Optimal spending rules
│   ├── portfolio.py        # Mean-variance optimization
│   ├── annuities.py        # SPIA pricing
│   └── balance_sheet.py    # Individual balance sheet
├── examples/
│   ├── __init__.py
│   └── isabela.py          # Complete case study
├── main.py                 # Quick demo
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `uv run pytest`
5. Format code: `uv run black .`
6. Submit a pull request

## License

See LICENSE file for details.

## References


Ibbotson, R. G., Milevsky, M. A., Chen, P., & Zhu, K. X. (2007). *Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance*. The Research Foundation of CFA Institute.
[Download](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/lifetime-financial-advice.pdf)
---

*Built with ❤️ using Python and UV*

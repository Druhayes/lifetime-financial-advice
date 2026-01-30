"""
Lifecycle Financial Planning Library

A Python implementation of lifecycle financial planning models based on
"Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
by Ibbotson, Milevsky, Chen, and Zhu.

Modules:
    mortality: Gompertz survival probability calculations
    utility: CRRA and Levy-Markowitz utility functions
    present_value: Present value and human capital calculations
    income: Wage curves and income modeling
    spending: Optimal spending and consumption rules
    portfolio: Mean-variance optimization and asset allocation
    annuities: SPIA pricing and mortality credits
    balance_sheet: Individual economic balance sheet
"""

# Import modules for use as `from lifecycle_planning.core import present_value`
from . import (
    annuities,
    balance_sheet,
    income,
    mortality,
    portfolio,
    present_value,
    spending,
    utility,
)

# Also import individual functions for convenience
from .annuities import (
    annuity_adjusted_survival,
    iva_units,
    mortality_credit,
    spia_price,
)
from .balance_sheet import (
    FinancialPreferences,
    IndividualBalanceSheet,
)
from .income import (
    WAGE_CURVES,
    income_curve,
    income_multiplier,
    projected_income,
    risky_income,
)
from .mortality import (
    GOMPERTZ_PARAMS,
    death_probability,
    gompertz_survival_prob,
    joint_survival_probability,
    life_expectancy,
    survival_probability,
    truncated_gompertz,
)
from .portfolio import (
    certainty_equivalent_return,
    mvo_optimal_portfolio,
    portfolio_expected_return,
    portfolio_std,
    portfolio_variance,
    theta_to_lambda,
)
from .present_value import (
    align_income_survival,
    human_capital,
    human_capital_mortality_weighted,
    income_discount_rate,
    liability_discount_rate,
    liability_value,
    liability_value_mortality_weighted,
    net_worth,
    pv,
    present_value_growing,
)
from .spending import (
    consumption_growth_rate,
    discretionary_consumption,
    optimal_consumption_divisor,
    rescheduling_factor,
    spending_rule_no_annuities,
    spending_rule_with_annuities,
)
from .utility import (
    crra_utility,
    levy_markowitz_utility,
    marginal_utility,
    risk_adjusted_expected_return,
)

__version__ = "0.1.0"
__all__ = [
    # modules (for imports like: from core import present_value)
    "annuities",
    "balance_sheet",
    "income",
    "mortality",
    "portfolio",
    "present_value",
    "spending",
    "utility",
    # mortality
    "gompertz_survival_prob",
    "truncated_gompertz",
    "survival_probability",
    "joint_survival_probability",
    "death_probability",
    "life_expectancy",
    "GOMPERTZ_PARAMS",
    # utility
    "crra_utility",
    "marginal_utility",
    "levy_markowitz_utility",
    "risk_adjusted_expected_return",
    # present_value (functions)
    "pv",
    "present_value_growing",
    "align_income_survival",
    "human_capital",
    "human_capital_mortality_weighted",
    "income_discount_rate",
    "liability_discount_rate",
    "liability_value",
    "liability_value_mortality_weighted",
    "net_worth",
    # income
    "income_curve",
    "income_multiplier",
    "projected_income",
    "risky_income",
    "WAGE_CURVES",
    # spending
    "consumption_growth_rate",
    "optimal_consumption_divisor",
    "discretionary_consumption",
    "rescheduling_factor",
    "spending_rule_no_annuities",
    "spending_rule_with_annuities",
    # portfolio
    "portfolio_expected_return",
    "portfolio_variance",
    "portfolio_std",
    "mvo_optimal_portfolio",
    "theta_to_lambda",
    "certainty_equivalent_return",
    # annuities
    "spia_price",
    "mortality_credit",
    "iva_units",
    "annuity_adjusted_survival",
    # balance_sheet
    "IndividualBalanceSheet",
    "FinancialPreferences",
]

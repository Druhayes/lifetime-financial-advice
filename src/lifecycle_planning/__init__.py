"""
Lifecycle Financial Planning Library

A Python implementation of lifecycle financial planning models based on
"Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
by Ibbotson, Milevsky, Chen, and Zhu.

This library provides tools for:
- Mortality modeling using Gompertz survival probabilities
- CRRA utility and preference modeling
- Human capital and liability calculations
- Optimal spending rules for retirement planning
- Mean-variance portfolio optimization
- Annuity pricing and analysis
- Individual economic balance sheets for comprehensive financial planning

Example:
    >>> from lifecycle_planning import IndividualBalanceSheet, FinancialPreferences
    >>>
    >>> prefs = FinancialPreferences(theta=0.35, rho=0.02, eta=0.50)
    >>> client = IndividualBalanceSheet(
    ...     current_age=30,
    ...     retirement_age=65,
    ...     gender="female",
    ...     current_income=85000,
    ...     nondiscretionary_spending=50000,
    ...     taxable_wealth=25000,
    ...     tax_advantaged_wealth=150000,
    ...     preferences=prefs,
    ... )
    >>> print(f"Net Worth: ${client.net_worth:,.0f}")
"""

from lifecycle_planning._version import __version__

# Import from core modules
from lifecycle_planning.core import (
    GOMPERTZ_PARAMS,
    WAGE_CURVES,
    FinancialPreferences,
    # balance_sheet
    IndividualBalanceSheet,
    annuity_adjusted_survival,
    certainty_equivalent_return,
    # spending
    consumption_growth_rate,
    # utility
    crra_utility,
    death_probability,
    discretionary_consumption,
    # mortality
    gompertz_survival_prob,
    human_capital,
    human_capital_mortality_weighted,
    # income
    income_curve,
    income_discount_rate,
    income_multiplier,
    iva_units,
    joint_survival_probability,
    levy_markowitz_utility,
    liability_value,
    liability_value_mortality_weighted,
    life_expectancy,
    marginal_utility,
    mortality_credit,
    mvo_optimal_portfolio,
    net_worth,
    optimal_consumption_divisor,
    # portfolio
    portfolio_expected_return,
    portfolio_std,
    portfolio_variance,
    # present_value
    present_value,
    present_value_growing,
    projected_income,
    rescheduling_factor,
    risk_adjusted_expected_return,
    risky_income,
    spending_rule_no_annuities,
    spending_rule_with_annuities,
    # annuities
    spia_price,
    survival_probability,
    theta_to_lambda,
    truncated_gompertz,
)

__all__ = [
    "__version__",
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
    # present_value
    "present_value",
    "present_value_growing",
    "human_capital",
    "human_capital_mortality_weighted",
    "income_discount_rate",
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

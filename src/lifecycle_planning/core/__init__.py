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

from .mortality import (
    gompertz_survival_prob,
    truncated_gompertz,
    survival_probability,
    joint_survival_probability,
    death_probability,
    life_expectancy,
    GOMPERTZ_PARAMS,
)

from .utility import (
    crra_utility,
    marginal_utility,
    levy_markowitz_utility,
    risk_adjusted_expected_return,
)

from .present_value import (
    present_value,
    present_value_growing,
    human_capital,
    human_capital_mortality_weighted,
    liability_value,
    liability_value_mortality_weighted,
    net_worth,
)

from .income import (
    income_curve,
    income_multiplier,
    projected_income,
    risky_income,
    WAGE_CURVES,
)

from .spending import (
    consumption_growth_rate,
    optimal_consumption_divisor,
    discretionary_consumption,
    rescheduling_factor,
    spending_rule_no_annuities,
    spending_rule_with_annuities,
)

from .portfolio import (
    portfolio_expected_return,
    portfolio_variance,
    portfolio_std,
    mvo_optimal_portfolio,
    theta_to_lambda,
    certainty_equivalent_return,
)

from .annuities import (
    spia_price,
    mortality_credit,
    iva_units,
    annuity_adjusted_survival,
)

from .balance_sheet import (
    IndividualBalanceSheet,
    FinancialPreferences,
)

__version__ = "0.1.0"
__all__ = [
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

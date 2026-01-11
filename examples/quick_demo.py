#!/usr/bin/env python3
"""
Lifetime Financial Advice - Python Implementation

A library for lifecycle financial planning based on:
"Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
by Ibbotson, Milevsky, Chen, and Zhu.

This library provides functions for:
    - Mortality modeling (Gompertz survival probabilities)
    - CRRA utility and preference modeling
    - Human capital and liability calculations
    - Optimal spending rules
    - Mean-variance portfolio optimization
    - Annuity pricing and analysis
    - Individual economic balance sheets

Usage:
    python main.py              # Run quick demo
    python examples/isabela.py  # Run full Isabela case study
"""

import numpy as np


def demo():
    """Run a quick demonstration of the library."""
    print("=" * 70)
    print(" LIFETIME FINANCIAL ADVICE - LIBRARY DEMO")
    print("=" * 70)

    # Import library modules
    from lifecycle_planning import (
        # Mortality
        survival_probability,
        life_expectancy,
        death_probability,
        GOMPERTZ_PARAMS,
        # Utility
        crra_utility,
        levy_markowitz_utility,
        # Present value
        human_capital,
        present_value,
        net_worth,
        # Portfolio
        portfolio_expected_return,
        mvo_optimal_portfolio,
        certainty_equivalent_return,
        # Spending
        consumption_growth_rate,
        optimal_consumption_divisor,
        discretionary_consumption,
        # Annuities
        spia_price,
        mortality_credit,
        # Balance Sheet
        IndividualBalanceSheet,
        FinancialPreferences,
    )
    from lifecycle_planning.core.mortality import survival_probability_series

    # -------------------------------------------------------------------------
    # 1. Mortality Functions
    # -------------------------------------------------------------------------
    print("\n1. GOMPERTZ MORTALITY MODEL")
    print("-" * 40)

    # Survival probabilities
    prob_65 = survival_probability(25, 65, "female")
    prob_85 = survival_probability(25, 85, "female")
    le_25f = life_expectancy(25, "female")

    print(f"   25-year-old woman:")
    print(f"   - Probability of reaching 65: {prob_65:.1%}")
    print(f"   - Probability of reaching 85: {prob_85:.1%}")
    print(f"   - Life expectancy: {25 + le_25f:.1f} years")

    # -------------------------------------------------------------------------
    # 2. Utility Functions
    # -------------------------------------------------------------------------
    print("\n2. CRRA UTILITY FUNCTION")
    print("-" * 40)

    consumption = 50000
    theta = 0.35  # Risk tolerance

    utility = crra_utility(consumption, theta)
    print(f"   Consumption: ${consumption:,}")
    print(f"   Risk tolerance (θ): {theta}")
    print(f"   Utility: {utility:.2f}")

    # Levy-Markowitz utility for portfolio
    mu, sigma = 0.07, 0.15
    lm_utility = levy_markowitz_utility(mu, sigma, theta)
    print(f"\n   Portfolio (μ={mu:.0%}, σ={sigma:.0%}):")
    print(f"   Levy-Markowitz utility: {lm_utility:.4f}")

    # -------------------------------------------------------------------------
    # 3. Present Value & Human Capital
    # -------------------------------------------------------------------------
    print("\n3. HUMAN CAPITAL CALCULATION")
    print("-" * 40)

    income = np.array([75000] * 40)  # $75k for 40 years
    discount_rate = 0.03
    hc = human_capital(income, discount_rate)

    print(f"   Income: ${income[0]:,}/year for 40 years")
    print(f"   Discount rate: {discount_rate:.1%}")
    print(f"   Human capital (PV): ${hc:,.0f}")

    # -------------------------------------------------------------------------
    # 4. Portfolio Optimization
    # -------------------------------------------------------------------------
    print("\n4. MEAN-VARIANCE OPTIMIZATION")
    print("-" * 40)

    returns = np.array([0.07, 0.035])
    cov_matrix = np.array([
        [0.155**2, 0.155 * 0.045 * 0.15],
        [0.155 * 0.045 * 0.15, 0.045**2],
    ])

    weights, exp_ret, std = mvo_optimal_portfolio(returns, cov_matrix, theta)

    print(f"   Risk tolerance (θ): {theta}")
    print(f"   Optimal allocation:")
    print(f"   - Stocks: {weights[0]:.1%}")
    print(f"   - Bonds:  {weights[1]:.1%}")
    print(f"   Expected return: {exp_ret:.2%}")
    print(f"   Std deviation: {std:.2%}")

    # -------------------------------------------------------------------------
    # 5. Spending Rules
    # -------------------------------------------------------------------------
    print("\n5. OPTIMAL SPENDING RULE")
    print("-" * 40)

    nw = 1000000  # $1M net worth
    r = 0.025     # Risk-free rate
    rho = 0.02    # Impatience
    eta = 0.50    # EOIS
    years = 30

    g = consumption_growth_rate(r, rho, eta)
    divisor = optimal_consumption_divisor(years, r, rho, eta)
    consumption = discretionary_consumption(nw, divisor)

    print(f"   Net worth: ${nw:,}")
    print(f"   Consumption growth rate: {g:.2%}")
    print(f"   Consumption divisor: {divisor:.2f}")
    print(f"   Optimal spending: ${consumption:,.0f}/year")
    print(f"   Spending rate: {consumption/nw:.2%}")

    # -------------------------------------------------------------------------
    # 6. Annuity Pricing
    # -------------------------------------------------------------------------
    print("\n6. ANNUITY ANALYSIS")
    print("-" * 40)

    survival = survival_probability_series(65, 40, "female")
    price = spia_price(65, 0.025, survival, 10000)
    mc = mortality_credit(0.025, survival[10])  # 10 years out

    print(f"   SPIA for $10,000/year at age 65:")
    print(f"   - Price: ${price:,.0f}")
    print(f"   - Payout rate: {10000/price:.2%}")
    print(f"   Mortality credit at age 75: {mc:.2%}")

    # -------------------------------------------------------------------------
    # 7. Individual Balance Sheet
    # -------------------------------------------------------------------------
    print("\n7. INDIVIDUAL BALANCE SHEET")
    print("-" * 40)

    prefs = FinancialPreferences(theta=0.35, rho=0.02, eta=0.50)
    balance_sheet = IndividualBalanceSheet(
        current_age=30,
        retirement_age=65,
        gender="male",
        current_income=100000,
        nondiscretionary_spending=45000,
        taxable_wealth=50000,
        tax_advantaged_wealth=75000,
        preferences=prefs,
    )

    print(f"   30-year-old male, $100k income:")
    print(f"   - Financial wealth: ${balance_sheet.financial_wealth:,.0f}")
    print(f"   - Human capital: ${balance_sheet.human_capital_value:,.0f}")
    print(f"   - Liabilities: ${balance_sheet.liability_value:,.0f}")
    print(f"   - Net worth: ${balance_sheet.net_worth:,.0f}")

    spending = balance_sheet.optimal_spending()
    print(f"\n   Optimal spending: ${spending['total_consumption']:,.0f}/year")
    print(f"   Savings rate: {spending['savings_rate']:.1%}")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" For the full Isabela case study, run:")
    print("     python examples/isabela.py")
    print("=" * 70)


if __name__ == "__main__":
    demo()

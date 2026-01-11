"""
Isabela Case Study

Reproduces the Isabela example from "Lifetime Financial Advice".

Isabela's profile:
    - Age: 25 years old
    - Gender: Female
    - Education: Post-college
    - Current Income: $75,000
    - Nondiscretionary Spending: $40,000
    - Financial Wealth: $100,000 (tax-advantaged)
    - Risk Tolerance (θ): 35%
    - Impatience (ρ): 2%
    - EOIS (η): 50%
    - Intergenerational Elasticity (γ): 25%
    - Bequest Strength (φ): 1.5%
    - Life Expectancy: 94 (personalized due to family longevity)

This script demonstrates:
    1. Building Isabela's individual balance sheet
    2. Calculating optimal spending recommendations
    3. Projecting her balance sheet to various ages
    4. Analyzing different market scenarios
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lifecycle_planning.core.annuities import mortality_credit, spia_price
from lifecycle_planning.core.balance_sheet import (
    compare_scenarios,
    create_isabela,
)
from lifecycle_planning.core.mortality import (
    life_expectancy,
    survival_probability_series,
)
from lifecycle_planning.core.portfolio import mvo_optimal_portfolio


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run the Isabela case study."""

    print_section("ISABELA CASE STUDY")
    print("""
From "Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
by Ibbotson, Milevsky, Chen, and Zhu

Isabela is a 25-year-old woman with a post-college education who has
recently started working with Paula, her financial planner.
    """)

    # =========================================================================
    # 1. Create Isabela's Balance Sheet
    # =========================================================================
    print_section("1. INDIVIDUAL BALANCE SHEET")

    isabela = create_isabela()
    print(isabela.summary())

    # =========================================================================
    # 2. Financial Preferences
    # =========================================================================
    print_section("2. FINANCIAL PREFERENCES")

    prefs = isabela.preferences
    print(f"""
    Risk Tolerance (θ):                {prefs.theta:.0%}
        → Low: Isabela has a somewhat low tolerance for risk

    Impatience (ρ):                    {prefs.rho:.0%}
        → Patient: Willing to live frugally now for higher living standard later

    Preference for Smooth Consumption (η): {prefs.eta:.0%}
        → Moderate: Willing to have moderate interruptions to consumption

    Intergenerational Elasticity (γ):  {prefs.gamma:.0%}
        → Low: Low flexibility regarding consumption vs bequest trade-off

    Bequest Strength (φ):              {prefs.phi:.1%}
        → Moderate: Prefers moderate standard of living and moderate bequest
    """)

    # =========================================================================
    # 3. Personalized Mortality
    # =========================================================================
    print_section("3. PERSONALIZED MORTALITY (GOMPERTZ MODEL)")

    # Standard life expectancy for 25-year-old woman
    standard_le = life_expectancy(25, "female") + 25
    personalized_le = 94  # Isabela's personalized estimate

    print(f"""
    Isabela's family has exceptional longevity:
        - Paternal grandfather died at 99
        - Maternal grandparents alive at 92 and 94
        - Two paternal great-aunts alive at 100 and 102

    Standard life expectancy for 25-year-old woman: {standard_le:.1f} years
    Personalized life expectancy for Isabela:       {personalized_le:.1f} years

    This affects all calculations involving survival probability.
    """)

    # Show survival probability comparison
    survival_standard = survival_probability_series(25, 70, "female")
    survival_personalized = survival_probability_series(
        25, 70, "female", m_override=isabela.m_personalized
    )

    print("    Survival Probability Comparison:")
    print("    " + "-" * 50)
    print("    Age    Standard    Personalized    Difference")
    for age in [65, 75, 85, 95]:
        idx = age - 25
        if idx < len(survival_standard):
            std = survival_standard[idx]
            pers = survival_personalized[idx]
            diff = pers - std
            print(f"    {age}     {std:>8.1%}       {pers:>8.1%}        {diff:>+.1%}")

    # =========================================================================
    # 4. Optimal Spending
    # =========================================================================
    print_section("4. OPTIMAL SPENDING RECOMMENDATION")

    spending = isabela.optimal_spending()
    print(isabela.spending_summary())

    print(f"""
    Interpretation:
        - Isabela should consume ${spending["discretionary_consumption"]:,.0f} in discretionary spending
        - Plus ${spending["nondiscretionary_consumption"]:,.0f} in nondiscretionary spending
        - Total: ${spending["total_consumption"]:,.0f} per year

        - Consumption should grow at {spending["consumption_growth_rate"]:.2%} per year
          (This is because r > ρ, so patient investor defers consumption)

        - Implied savings rate: {spending["savings_rate"]:.1%} of income
    """)

    # =========================================================================
    # 5. Balance Sheet Projections
    # =========================================================================
    print_section("5. BALANCE SHEET PROJECTIONS")

    projection_ages = [45, 65, 85]

    for target_age in projection_ages:
        print(f"\n    Projection to Age {target_age}:")
        print("    " + "-" * 50)

        scenarios = compare_scenarios(isabela, target_age)

        print("                  Optimistic      Median    Pessimistic")
        for metric in ["financial_wealth", "human_capital", "liabilities", "net_worth"]:
            label = metric.replace("_", " ").title()
            opt = scenarios["optimistic"][metric]
            med = scenarios["median"][metric]
            pes = scenarios["pessimistic"][metric]
            print(f"    {label:20s} ${opt:>10,.0f} ${med:>10,.0f} ${pes:>10,.0f}")

    # =========================================================================
    # 6. Portfolio Optimization
    # =========================================================================
    print_section("6. OPTIMAL PORTFOLIO (MEAN-VARIANCE)")

    # Simple 2-asset case: Stocks and Bonds
    returns = np.array([0.07, 0.035])  # Equity, Bonds
    std_devs = np.array([0.155, 0.045])
    correlation = 0.15
    cov_matrix = np.array(
        [
            [std_devs[0] ** 2, std_devs[0] * std_devs[1] * correlation],
            [std_devs[0] * std_devs[1] * correlation, std_devs[1] ** 2],
        ]
    )

    # Find Isabela's optimal portfolio
    weights, exp_ret, port_std = mvo_optimal_portfolio(
        returns, cov_matrix, isabela.preferences.theta
    )

    print(f"""
    Given Isabela's risk tolerance (θ = {isabela.preferences.theta:.0%}):

    Optimal Asset Allocation:
        Equities:    {weights[0]:>6.1%}
        Bonds:       {weights[1]:>6.1%}

    Expected Portfolio Return:     {exp_ret:>6.2%}
    Portfolio Standard Deviation:  {port_std:>6.2%}

    Note: With human capital being mostly bond-like (80% fixed income),
    Isabela's financial wealth can be more aggressively invested.
    """)

    # =========================================================================
    # 7. Annuity Analysis
    # =========================================================================
    print_section("7. ANNUITY VALUE AT RETIREMENT")

    # Calculate at age 65
    retirement_survival = survival_probability_series(65, 40, "female")

    # SPIA price for $1 annual income
    spia_1dollar = spia_price(65, 0.025, retirement_survival, 1.0)
    payout_rate = 1 / spia_1dollar

    # Mortality credit at various ages
    mc_65 = mortality_credit(0.025, retirement_survival[1])
    mc_75 = mortality_credit(0.025, retirement_survival[11])
    mc_85 = mortality_credit(0.025, retirement_survival[21])

    print(f"""
    SPIA (Single Premium Immediate Annuity) Analysis at Age 65:

    Price of $1/year lifetime income:  ${spia_1dollar:,.2f}
    Implied payout rate:               {payout_rate:.2%}

    Mortality Credits by Age:
        Age 65-66:  {mc_65:>6.2%}
        Age 75-76:  {mc_75:>6.2%}
        Age 85-86:  {mc_85:>6.2%}

    Mortality credits increase with age as survival probability decreases.
    This makes annuities increasingly valuable for longevity insurance.
    """)

    # =========================================================================
    # 8. Human Capital and Wage Projection
    # =========================================================================
    print_section("8. HUMAN CAPITAL AND WAGE PROJECTION")

    income_projection = isabela.income_stream[:40]  # Working years

    print("    Projected Income by Age:")
    print("    " + "-" * 50)
    for i, age in enumerate([25, 35, 45, 55, 64]):
        idx = age - 25
        if idx < len(income_projection):
            income = income_projection[idx]
            print(f"    Age {age}:  ${income:>12,.0f}")

    print(f"""
    Human Capital (PV of future income): ${isabela.human_capital_value:>12,.0f}

    This value accounts for:
        - Wage growth based on education and gender curves
        - Income multiplier (Isabela earns above curve average)
        - Mortality weighting (personalized survival probability)
        - Discount rate reflecting income risk (20% equity-like)
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")

    print(f"""
    Key Takeaways for Isabela:

    1. NET WORTH: ${isabela.net_worth:>,.0f}
       - Much higher than financial wealth alone (${isabela.financial_wealth:,.0f})
       - Human capital is her largest asset

    2. OPTIMAL SPENDING: ${spending["total_consumption"]:>,.0f}/year
       - Save {spending["savings_rate"]:.0%} of income
       - Consumption grows at {spending["consumption_growth_rate"]:.2%}/year

    3. ASSET ALLOCATION: {weights[0]:.0%} Equities / {weights[1]:.0%} Bonds
       - Can take more financial risk since human capital is bond-like

    4. LONGEVITY RISK:
       - Personalized life expectancy extends planning horizon
       - Consider annuities at retirement for longevity insurance

    5. PROJECTIONS:
       - Median net worth at age 65: ${scenarios["median"]["net_worth"]:,.0f}
       - Range: ${scenarios["pessimistic"]["net_worth"]:,.0f} to ${scenarios["optimistic"]["net_worth"]:,.0f}
    """)


if __name__ == "__main__":
    main()

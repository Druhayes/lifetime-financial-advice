"""
Individual Balance Sheet for Lifecycle Financial Planning

Implements the individual economic balance sheet concept:
    - Financial wealth (taxable + tax-advantaged)
    - Human capital (present value of future income)
    - Liabilities (present value of nondiscretionary consumption)
    - Net worth (available for discretionary consumption)

Based on Chapters 2 and 12 of "Lifetime Financial Advice".

The balance sheet provides a holistic view of an individual's
financial situation, including both financial and human capital.

References:
    - Ibbotson, R., Milevsky, M., Chen, P., & Zhu, K.
      "Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from .mortality import (
    survival_probability_series,
    death_probability_series,
    personalized_mode,
    GOMPERTZ_PARAMS,
)
from .present_value import (
    human_capital_mortality_weighted,
    liability_value_mortality_weighted,
    income_discount_rate,
    liability_discount_rate,
)
from .income import projected_income, full_income_stream
from .spending import (
    consumption_growth_rate,
    optimal_consumption_divisor_with_annuities,
    discretionary_consumption,
)
from .utility import PreferenceParams


@dataclass
class FinancialPreferences:
    """
    Financial preferences for lifecycle planning.

    Attributes:
        theta: Risk tolerance (0-1), e.g., 0.35 for moderate
        rho: Impatience / subjective discount rate, e.g., 0.02
        eta: EOIS (preference for smooth consumption), e.g., 0.5
        gamma: Intergenerational elasticity (bequest flexibility), e.g., 0.25
        phi: Strength of bequest motive, e.g., 1.5
    """
    theta: float = 0.35  # Risk tolerance
    rho: float = 0.02    # Impatience
    eta: float = 0.50    # EOIS
    gamma: float = 0.25  # Intergenerational elasticity
    phi: float = 1.5     # Bequest strength

    def to_preference_params(self) -> PreferenceParams:
        """Convert to PreferenceParams dataclass."""
        return PreferenceParams(
            theta=self.theta,
            rho=self.rho,
            eta=self.eta,
            gamma=self.gamma,
            phi=self.phi,
        )


@dataclass
class IndividualBalanceSheet:
    """
    Individual Economic Balance Sheet.

    Provides a holistic view of an individual's financial situation,
    combining financial assets with human capital and liabilities.

    Example usage:
        >>> prefs = FinancialPreferences(theta=0.35, rho=0.02, eta=0.5)
        >>> bs = IndividualBalanceSheet(
        ...     current_age=25,
        ...     retirement_age=65,
        ...     gender="female",
        ...     current_income=75000,
        ...     nondiscretionary_spending=40000,
        ...     taxable_wealth=0,
        ...     tax_advantaged_wealth=100000,
        ...     preferences=prefs,
        ... )
        >>> print(f"Net Worth: ${bs.net_worth:,.0f}")
        Net Worth: $1,475,625
    """

    # Demographics
    current_age: int
    retirement_age: int = 65
    max_age: int = 120
    gender: str = "female"
    education: str = "post_college"

    # Current financial situation
    current_income: float = 75000
    nondiscretionary_spending: float = 40000
    taxable_wealth: float = 0
    tax_advantaged_wealth: float = 100000

    # Social Security
    social_security_monthly: float = 2000

    # Risk assumptions
    risk_free_rate: float = 0.025
    equity_return: float = 0.07
    hc_equity_weight: float = 0.20  # Human capital equity correlation
    liability_equity_weight: float = 0.15  # Liability equity correlation

    # Preferences
    preferences: FinancialPreferences = field(default_factory=FinancialPreferences)

    # Personalization
    life_expectancy_override: Optional[float] = None

    # Computed fields (set in __post_init__)
    survival_probs: np.ndarray = field(default=None, repr=False)
    death_probs: np.ndarray = field(default=None, repr=False)
    income_stream: np.ndarray = field(default=None, repr=False)
    consumption_stream: np.ndarray = field(default=None, repr=False)
    human_capital_value: float = field(default=0.0)
    liability_value: float = field(default=0.0)
    financial_wealth: float = field(default=0.0)
    net_worth: float = field(default=0.0)
    m_personalized: float = field(default=None)

    def __post_init__(self):
        """Calculate all derived values."""
        self._calculate_mortality()
        self._calculate_income_and_consumption()
        self._calculate_values()

    def _calculate_mortality(self):
        """Calculate survival and death probabilities."""
        years = self.max_age - self.current_age

        # Get base Gompertz parameters
        params = GOMPERTZ_PARAMS[self.gender.lower()]
        m = params.m
        b = params.b

        # Personalize if life expectancy override provided
        if self.life_expectancy_override is not None:
            remaining_life = self.life_expectancy_override - self.current_age
            m = personalized_mode(remaining_life, self.current_age, self.gender)
            self.m_personalized = m

        self.survival_probs = survival_probability_series(
            self.current_age, years, self.gender, m, b
        )
        self.death_probs = death_probability_series(
            self.current_age, years, self.gender, m, b
        )

    def _calculate_income_and_consumption(self):
        """Calculate income and consumption streams."""
        years = self.max_age - self.current_age

        # Full income stream (wages + Social Security)
        self.income_stream = full_income_stream(
            self.current_age,
            self.retirement_age,
            self.max_age,
            self.current_income,
            self.social_security_monthly,
            self.gender,
            self.education,
        )

        # Consumption stream (constant nondiscretionary spending)
        self.consumption_stream = np.full(years, self.nondiscretionary_spending)

    def _calculate_values(self):
        """Calculate human capital, liabilities, and net worth."""
        # Discount rates
        hc_rate = income_discount_rate(
            self.risk_free_rate, self.equity_return, self.hc_equity_weight
        )
        liab_rate = liability_discount_rate(
            self.risk_free_rate, self.equity_return, self.liability_equity_weight
        )

        # Ensure arrays are same length
        min_len = min(
            len(self.income_stream),
            len(self.consumption_stream),
            len(self.survival_probs)
        )

        # Human capital (mortality-weighted)
        self.human_capital_value = human_capital_mortality_weighted(
            self.income_stream[:min_len],
            hc_rate,
            self.survival_probs[:min_len],
        )

        # Liabilities (mortality-weighted)
        self.liability_value = liability_value_mortality_weighted(
            self.consumption_stream[:min_len],
            liab_rate,
            self.survival_probs[:min_len],
        )

        # Financial wealth
        self.financial_wealth = self.taxable_wealth + self.tax_advantaged_wealth

        # Net worth
        self.net_worth = (
            self.financial_wealth
            + self.human_capital_value
            - self.liability_value
        )

    def get_balance_sheet_dict(self) -> Dict:
        """Return balance sheet as dictionary."""
        return {
            "Assets": {
                "Financial Wealth": {
                    "Taxable": self.taxable_wealth,
                    "Tax-Advantaged": self.tax_advantaged_wealth,
                    "Total": self.financial_wealth,
                },
                "Human Capital": self.human_capital_value,
                "Total Assets": self.financial_wealth + self.human_capital_value,
            },
            "Liabilities": {
                "Nondiscretionary Consumption": self.liability_value,
                "Total Liabilities": self.liability_value,
            },
            "Net Worth": self.net_worth,
        }

    def optimal_spending(self, annuities_available: bool = True) -> Dict:
        """
        Calculate optimal spending based on current net worth.

        Args:
            annuities_available: Whether annuities are available at retirement

        Returns:
            Dictionary with spending information
        """
        years = self.max_age - self.current_age
        retirement_year = self.retirement_age - self.current_age

        # Consumption growth rate
        g = consumption_growth_rate(
            self.risk_free_rate,
            self.preferences.rho,
            self.preferences.eta,
        )

        # Consumption divisor
        divisor = optimal_consumption_divisor_with_annuities(
            years,
            self.risk_free_rate,
            self.preferences.rho,
            self.preferences.eta,
            self.survival_probs,
            retirement_year,
        )

        # Optimal discretionary consumption
        disc_consumption = discretionary_consumption(self.net_worth, divisor)

        # Total consumption
        total_consumption = disc_consumption + self.nondiscretionary_spending

        return {
            "discretionary_consumption": disc_consumption,
            "nondiscretionary_consumption": self.nondiscretionary_spending,
            "total_consumption": total_consumption,
            "consumption_growth_rate": g,
            "consumption_divisor": divisor,
            "savings_rate": max(0, 1 - total_consumption / self.current_income),
        }

    def project_balance_sheet(
        self,
        target_age: int,
        scenario: str = "median",
    ) -> "IndividualBalanceSheet":
        """
        Project balance sheet to future age.

        Args:
            target_age: Age to project to
            scenario: "optimistic", "median", or "pessimistic"

        Returns:
            Projected IndividualBalanceSheet
        """
        if target_age <= self.current_age:
            raise ValueError("target_age must be greater than current_age")

        years_forward = target_age - self.current_age

        # Scenario return adjustments
        scenario_returns = {
            "optimistic": 0.02,
            "median": 0.0,
            "pessimistic": -0.02,
        }
        return_adjustment = scenario_returns.get(scenario, 0.0)

        # Project financial wealth with investment returns
        annual_return = self.risk_free_rate + return_adjustment
        if scenario == "optimistic":
            annual_return = self.equity_return + return_adjustment
        elif scenario == "pessimistic":
            annual_return = self.risk_free_rate + return_adjustment

        # Simple projection: wealth grows at return, minus spending
        spending = self.optimal_spending()
        annual_savings = self.current_income - spending["total_consumption"]

        # Project wealth (simplified)
        projected_wealth = self.financial_wealth
        for _ in range(years_forward):
            projected_wealth = projected_wealth * (1 + annual_return) + annual_savings

        # Update income (use projected income at target age)
        projected_income = 0
        if target_age < self.retirement_age:
            idx = target_age - self.current_age
            if idx < len(self.income_stream):
                projected_income = self.income_stream[idx]

        # Create new balance sheet at projected age
        projected_bs = IndividualBalanceSheet(
            current_age=target_age,
            retirement_age=self.retirement_age,
            max_age=self.max_age,
            gender=self.gender,
            education=self.education,
            current_income=projected_income if projected_income > 0 else self.social_security_monthly * 12,
            nondiscretionary_spending=self.nondiscretionary_spending,
            taxable_wealth=projected_wealth * 0.3,  # Simplified split
            tax_advantaged_wealth=projected_wealth * 0.7,
            social_security_monthly=self.social_security_monthly,
            risk_free_rate=self.risk_free_rate,
            equity_return=self.equity_return,
            preferences=self.preferences,
            life_expectancy_override=self.life_expectancy_override,
        )

        return projected_bs

    def summary(self) -> str:
        """Return formatted summary of balance sheet."""
        lines = [
            "=" * 60,
            "INDIVIDUAL ECONOMIC BALANCE SHEET",
            "=" * 60,
            f"Age: {self.current_age} | Gender: {self.gender.capitalize()} | Education: {self.education}",
            "",
            "ASSETS",
            "-" * 30,
            f"  Taxable Wealth:        ${self.taxable_wealth:>12,.0f}",
            f"  Tax-Advantaged:        ${self.tax_advantaged_wealth:>12,.0f}",
            f"  Financial Wealth:      ${self.financial_wealth:>12,.0f}",
            f"  Human Capital:         ${self.human_capital_value:>12,.0f}",
            f"  TOTAL ASSETS:          ${self.financial_wealth + self.human_capital_value:>12,.0f}",
            "",
            "LIABILITIES",
            "-" * 30,
            f"  Nondiscretionary:      ${self.liability_value:>12,.0f}",
            f"  TOTAL LIABILITIES:     ${self.liability_value:>12,.0f}",
            "",
            "=" * 60,
            f"  NET WORTH:             ${self.net_worth:>12,.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def spending_summary(self) -> str:
        """Return formatted spending recommendation."""
        spending = self.optimal_spending()
        lines = [
            "",
            "OPTIMAL SPENDING",
            "-" * 30,
            f"  Discretionary:         ${spending['discretionary_consumption']:>12,.0f}",
            f"  Nondiscretionary:      ${spending['nondiscretionary_consumption']:>12,.0f}",
            f"  TOTAL CONSUMPTION:     ${spending['total_consumption']:>12,.0f}",
            "",
            f"  Growth Rate:           {spending['consumption_growth_rate']:>12.2%}",
            f"  Implied Savings Rate:  {spending['savings_rate']:>12.2%}",
        ]
        return "\n".join(lines)


def create_isabela() -> IndividualBalanceSheet:
    """
    Create the Isabela example from the document.

    Isabela is the example investor used throughout
    "Lifetime Financial Advice":
        - 25-year-old woman
        - Post-college education
        - $75,000 current income
        - $40,000 nondiscretionary spending
        - $100,000 in tax-advantaged savings
        - Long-lived family (life expectancy ~94)
        - Moderate risk tolerance (theta=35%)

    Returns:
        IndividualBalanceSheet for Isabela
    """
    prefs = FinancialPreferences(
        theta=0.35,   # Risk tolerance: Low
        rho=0.02,     # Impatience: Patient
        eta=0.50,     # EOIS: Moderate smoothing
        gamma=0.25,   # Intergenerational elasticity: Low
        phi=1.5,      # Bequest strength: Moderate
    )

    isabela = IndividualBalanceSheet(
        current_age=25,
        retirement_age=65,
        max_age=120,
        gender="female",
        education="post_college",
        current_income=75000,
        nondiscretionary_spending=40000,
        taxable_wealth=0,
        tax_advantaged_wealth=100000,
        social_security_monthly=2000,
        risk_free_rate=0.025,
        equity_return=0.07,
        hc_equity_weight=0.20,
        liability_equity_weight=0.15,
        preferences=prefs,
        life_expectancy_override=94,  # High longevity family
    )

    return isabela


def compare_scenarios(
    balance_sheet: IndividualBalanceSheet,
    target_age: int,
) -> Dict:
    """
    Compare balance sheet projections across scenarios.

    Args:
        balance_sheet: Current balance sheet
        target_age: Age to project to

    Returns:
        Dictionary with optimistic, median, pessimistic projections
    """
    scenarios = {}
    for scenario in ["optimistic", "median", "pessimistic"]:
        projected = balance_sheet.project_balance_sheet(target_age, scenario)
        scenarios[scenario] = {
            "financial_wealth": projected.financial_wealth,
            "human_capital": projected.human_capital_value,
            "liabilities": projected.liability_value,
            "net_worth": projected.net_worth,
        }

    return scenarios

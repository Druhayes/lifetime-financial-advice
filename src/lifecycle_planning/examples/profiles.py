"""
Example client profiles for lifecycle financial planning.

This module provides pre-configured client profiles that demonstrate
various lifecycle financial planning scenarios. These profiles can be
used as starting points for analysis or as examples of how to configure
the IndividualBalanceSheet class.

Example:
    >>> from lifecycle_planning.examples.profiles import create_isabela
    >>> isabela = create_isabela()
    >>> print(f"Isabela's net worth: ${isabela.net_worth:,.0f}")
"""

from lifecycle_planning.core.balance_sheet import (
    create_isabela,
    IndividualBalanceSheet,
    FinancialPreferences,
)


__all__ = [
    "create_isabela",
    "create_conservative_retiree",
    "create_aggressive_professional",
]


def create_conservative_retiree() -> IndividualBalanceSheet:
    """
    Create a conservative retiree profile.

    Profile characteristics:
        - 65-year-old male
        - Already retired (no labor income)
        - $60,000 annual nondiscretionary spending
        - $1.1M total wealth ($300K taxable, $800K tax-advantaged)
        - Lower risk tolerance (theta=15%)
        - Stronger bequest motive

    Returns:
        IndividualBalanceSheet for conservative retiree
    """
    prefs = FinancialPreferences(
        theta=0.15,   # Lower risk tolerance
        rho=0.04,     # Higher impatience (shorter time horizon)
        eta=0.75,     # Prefers smoother consumption
        gamma=0.20,   # Less flexible on bequests
        phi=2.0,      # Stronger bequest motive
    )

    retiree = IndividualBalanceSheet(
        current_age=65,
        retirement_age=65,  # Already retired
        max_age=120,
        gender="male",
        education="college",
        current_income=0,   # No labor income
        nondiscretionary_spending=60000,
        taxable_wealth=300000,
        tax_advantaged_wealth=800000,
        social_security_monthly=3000,
        risk_free_rate=0.025,
        equity_return=0.07,
        hc_equity_weight=0.20,
        liability_equity_weight=0.15,
        preferences=prefs,
    )

    return retiree


def create_aggressive_professional() -> IndividualBalanceSheet:
    """
    Create an aggressive young professional profile.

    Profile characteristics:
        - 28-year-old female
        - High income ($120,000/year)
        - $45,000 annual nondiscretionary spending
        - $80K total wealth ($15K taxable, $65K tax-advantaged)
        - Higher risk tolerance (theta=55%)
        - Weaker bequest motive (focused on own consumption)

    Returns:
        IndividualBalanceSheet for aggressive professional
    """
    prefs = FinancialPreferences(
        theta=0.55,   # Higher risk tolerance
        rho=0.01,     # Lower impatience (longer time horizon)
        eta=0.30,     # More willing to vary consumption
        gamma=0.35,   # More flexible on bequests
        phi=0.8,      # Weaker bequest motive
    )

    young_prof = IndividualBalanceSheet(
        current_age=28,
        retirement_age=67,
        max_age=120,
        gender="female",
        education="post_college",
        current_income=120000,
        nondiscretionary_spending=45000,
        taxable_wealth=15000,
        tax_advantaged_wealth=65000,
        social_security_monthly=2500,
        risk_free_rate=0.025,
        equity_return=0.07,
        hc_equity_weight=0.20,
        liability_equity_weight=0.15,
        preferences=prefs,
    )

    return young_prof


# Re-export create_isabela for convenience
# (already defined in balance_sheet.py)
__all__ += ['create_isabela']

"""
Pytest configuration and shared fixtures for lifecycle-planning tests.

This module provides reusable fixtures for testing the lifecycle planning library.
"""

import pytest
import numpy as np
from lifecycle_planning import (
    FinancialPreferences,
    IndividualBalanceSheet,
)


@pytest.fixture
def isabela_preferences():
    """
    Financial preferences for Isabela example.

    Returns moderate risk tolerance with patient time horizon.
    """
    return FinancialPreferences(
        theta=0.35,   # Risk tolerance: Moderate
        rho=0.02,     # Impatience: Patient
        eta=0.50,     # EOIS: Moderate smoothing
        gamma=0.25,   # Intergenerational elasticity: Low
        phi=1.5,      # Bequest strength: Moderate
    )


@pytest.fixture
def conservative_preferences():
    """
    Financial preferences for conservative investor.

    Returns low risk tolerance with shorter time horizon.
    """
    return FinancialPreferences(
        theta=0.15,   # Risk tolerance: Conservative
        rho=0.04,     # Impatience: Higher (shorter horizon)
        eta=0.75,     # EOIS: Prefers smooth consumption
        gamma=0.20,   # Intergenerational elasticity: Low
        phi=2.0,      # Bequest strength: Strong
    )


@pytest.fixture
def aggressive_preferences():
    """
    Financial preferences for aggressive investor.

    Returns high risk tolerance with long time horizon.
    """
    return FinancialPreferences(
        theta=0.55,   # Risk tolerance: Aggressive
        rho=0.01,     # Impatience: Very patient
        eta=0.30,     # EOIS: Willing to vary consumption
        gamma=0.35,   # Intergenerational elasticity: Moderate
        phi=0.8,      # Bequest strength: Weak
    )


@pytest.fixture
def sample_young_client(isabela_preferences):
    """
    Sample young professional client for testing.

    Returns:
        IndividualBalanceSheet for 25-year-old female with moderate wealth
    """
    return IndividualBalanceSheet(
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
        preferences=isabela_preferences,
    )


@pytest.fixture
def sample_retiree(conservative_preferences):
    """
    Sample retiree client for testing.

    Returns:
        IndividualBalanceSheet for 65-year-old male retiree
    """
    return IndividualBalanceSheet(
        current_age=65,
        retirement_age=65,
        max_age=120,
        gender="male",
        education="college",
        current_income=0,  # Retired
        nondiscretionary_spending=60000,
        taxable_wealth=300000,
        tax_advantaged_wealth=800000,
        social_security_monthly=3000,
        risk_free_rate=0.025,
        equity_return=0.07,
        hc_equity_weight=0.20,
        liability_equity_weight=0.15,
        preferences=conservative_preferences,
    )


@pytest.fixture
def sample_portfolio_data():
    """
    Sample portfolio data for testing MVO.

    Returns:
        Dictionary with returns and covariance matrix for 2-asset portfolio
    """
    returns = np.array([0.07, 0.03])  # Stocks 7%, Bonds 3%

    # Covariance matrix
    # Stocks: 15% vol, Bonds: 5% vol, Correlation: 0.2
    cov_matrix = np.array([
        [0.15**2, 0.15 * 0.05 * 0.2],
        [0.15 * 0.05 * 0.2, 0.05**2]
    ])

    return {
        'returns': returns,
        'covariance': cov_matrix,
        'stock_return': 0.07,
        'bond_return': 0.03,
        'stock_vol': 0.15,
        'bond_vol': 0.05,
        'correlation': 0.2,
    }


@pytest.fixture
def sample_mortality_params():
    """
    Sample mortality parameters for testing.

    Returns:
        Dictionary with common test parameters
    """
    return {
        'young_age': 25,
        'middle_age': 45,
        'retirement_age': 65,
        'old_age': 85,
        'max_age': 120,
        'female': 'female',
        'male': 'male',
    }

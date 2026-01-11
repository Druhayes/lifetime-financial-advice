"""
Tests for lifecycle_planning.core.utility module.

Tests CRRA utility functions, Levy-Markowitz utility, and
preference parameter handling.
"""

import pytest
import numpy as np
from lifecycle_planning.core.utility import (
    crra_utility,
    marginal_utility,
    levy_markowitz_utility,
    risk_adjusted_expected_return,
    PreferenceParams,
)
from lifecycle_planning import FinancialPreferences


class TestCRRAUtility:
    """Test the CRRA (Constant Relative Risk Aversion) utility function."""

    def test_utility_increases_with_consumption(self):
        """Utility should increase as consumption increases."""
        theta = 0.35
        u_30k = crra_utility(30000, theta)
        u_50k = crra_utility(50000, theta)
        u_70k = crra_utility(70000, theta)

        assert u_30k < u_50k < u_70k

    def test_utility_concave(self):
        """Utility should be concave (diminishing marginal utility)."""
        theta = 0.35
        # First $20k increase
        increase_1 = crra_utility(40000, theta) - crra_utility(20000, theta)
        # Second $20k increase
        increase_2 = crra_utility(60000, theta) - crra_utility(40000, theta)

        # Second increase should give less additional utility
        assert increase_2 < increase_1

    def test_higher_theta_gives_higher_utility(self):
        """Higher risk tolerance (theta) gives higher utility for same consumption."""
        consumption = 50000
        u_low_theta = crra_utility(consumption, 0.2)
        u_high_theta = crra_utility(consumption, 0.5)

        # Higher theta → higher utility for positive consumption levels
        assert u_high_theta > u_low_theta

    def test_zero_consumption_undefined(self):
        """Zero consumption should raise error or return very negative value."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            crra_utility(0, 0.35)

    def test_negative_consumption_invalid(self):
        """Negative consumption should raise error."""
        with pytest.raises((ValueError, RuntimeWarning)):
            crra_utility(-10000, 0.35)

    def test_utility_realistic_values(self):
        """Utility should return reasonable values for typical consumption."""
        u = crra_utility(50000, 0.35)
        # Should be a reasonable positive number
        assert 100 < u < 200


class TestMarginalUtility:
    """Test the marginal utility function."""

    def test_marginal_utility_decreases(self):
        """Marginal utility should decrease with consumption."""
        theta = 0.35
        mu_20k = marginal_utility(20000, theta)
        mu_50k = marginal_utility(50000, theta)
        mu_80k = marginal_utility(80000, theta)

        assert mu_20k > mu_50k > mu_80k

    def test_marginal_utility_positive(self):
        """Marginal utility should always be positive."""
        theta = 0.35
        for consumption in [10000, 30000, 50000, 100000]:
            mu = marginal_utility(consumption, theta)
            assert mu > 0

    def test_higher_theta_higher_marginal_utility(self):
        """Higher theta gives higher marginal utility at same consumption."""
        consumption = 50000
        mu_low_theta = marginal_utility(consumption, 0.2)
        mu_high_theta = marginal_utility(consumption, 0.5)

        # MU = x^(θ-1), so higher θ means higher MU
        assert mu_high_theta > mu_low_theta


class TestLevyMarkowitzUtility:
    """Test the Levy-Markowitz utility approximation for portfolios."""

    def test_utility_increases_with_return(self):
        """Utility should increase as expected return increases."""
        theta = 0.35
        std = 0.15

        u_5pct = levy_markowitz_utility(0.05, std, theta)
        u_7pct = levy_markowitz_utility(0.07, std, theta)
        u_9pct = levy_markowitz_utility(0.09, std, theta)

        assert u_5pct < u_7pct < u_9pct

    def test_utility_decreases_with_volatility(self):
        """Utility should decrease as volatility increases."""
        theta = 0.35
        expected_return = 0.07

        u_vol_10 = levy_markowitz_utility(expected_return, 0.10, theta)
        u_vol_15 = levy_markowitz_utility(expected_return, 0.15, theta)
        u_vol_20 = levy_markowitz_utility(expected_return, 0.20, theta)

        assert u_vol_10 > u_vol_15 > u_vol_20

    def test_lower_theta_more_risk_averse(self):
        """Lower theta (more risk averse) should penalize volatility more."""
        expected_return = 0.07
        std = 0.20

        # Conservative investor (low theta)
        u_conservative = levy_markowitz_utility(expected_return, std, 0.15)
        # Aggressive investor (high theta)
        u_aggressive = levy_markowitz_utility(expected_return, std, 0.55)

        # Conservative should have lower utility due to high volatility penalty
        assert u_conservative < u_aggressive

    def test_zero_volatility_equals_crra_utility(self):
        """With zero volatility, utility should equal CRRA utility of (1+μ)."""
        expected_return = 0.06
        theta = 0.35
        u = levy_markowitz_utility(expected_return, 0.0, theta)
        expected_u = crra_utility(1 + expected_return, theta)
        assert u == pytest.approx(expected_u)

    def test_utility_can_be_negative(self):
        """High volatility with low return can give negative utility."""
        # Very high volatility, low return, risk averse
        u = levy_markowitz_utility(0.02, 0.30, 0.20)
        assert u < 0


class TestRiskAdjustedExpectedReturn:
    """Test risk-adjusted expected return calculation."""

    def test_risk_adjustment_lowers_return(self):
        """Risk adjustment should lower expected return for risky assets."""
        expected_return = 0.08
        std = 0.15
        lambda_ = 2.0  # Risk aversion parameter

        risk_adjusted = risk_adjusted_expected_return(
            expected_return, std, lambda_
        )

        # Risk adjustment should penalize volatility
        assert risk_adjusted < expected_return

    def test_zero_volatility_no_adjustment(self):
        """Zero volatility should give no risk adjustment."""
        expected_return = 0.05
        risk_adjusted = risk_adjusted_expected_return(
            expected_return, 0.0, 2.0
        )
        assert risk_adjusted == pytest.approx(expected_return)

    def test_higher_lambda_more_adjustment(self):
        """Higher lambda (more risk averse) should apply larger penalty."""
        expected_return = 0.08
        std = 0.15

        adj_low_lambda = risk_adjusted_expected_return(
            expected_return, std, 1.0
        )
        adj_high_lambda = risk_adjusted_expected_return(
            expected_return, std, 3.0
        )

        # Higher lambda applies bigger penalty
        assert adj_high_lambda < adj_low_lambda
        # Both should be below expected return
        assert adj_low_lambda < expected_return
        assert adj_high_lambda < expected_return


class TestPreferenceParams:
    """Test the PreferenceParams dataclass."""

    def test_preference_params_direct_creation(self):
        """Test creating PreferenceParams directly."""
        params = PreferenceParams(
            theta=0.40, rho=0.025, eta=0.60, gamma=0.30, phi=2.0
        )

        assert params.theta == 0.40
        assert params.rho == 0.025
        assert params.eta == 0.60
        assert params.gamma == 0.30
        assert params.phi == 2.0


class TestFinancialPreferences:
    """Test the FinancialPreferences dataclass."""

    def test_preferences_creation(self, isabela_preferences):
        """Test creating financial preferences."""
        assert isabela_preferences.theta == 0.35
        assert isabela_preferences.rho == 0.02
        assert isabela_preferences.eta == 0.50
        assert isabela_preferences.gamma == 0.25
        assert isabela_preferences.phi == 1.5

    def test_conservative_preferences(self, conservative_preferences):
        """Test conservative preferences have expected values."""
        # Conservative should have low theta (risk tolerance)
        assert conservative_preferences.theta < 0.25
        # Higher impatience (shorter horizon)
        assert conservative_preferences.rho > 0.03
        # Prefers smooth consumption
        assert conservative_preferences.eta > 0.60
        # Strong bequest motive
        assert conservative_preferences.phi > 1.5

    def test_aggressive_preferences(self, aggressive_preferences):
        """Test aggressive preferences have expected values."""
        # Aggressive should have high theta
        assert aggressive_preferences.theta > 0.45
        # Low impatience (long horizon)
        assert aggressive_preferences.rho < 0.02
        # Willing to vary consumption
        assert aggressive_preferences.eta < 0.40
        # Weak bequest motive
        assert aggressive_preferences.phi < 1.0

    def test_preferences_bounds(self):
        """Test that preference parameters are in valid ranges."""
        prefs = FinancialPreferences(
            theta=0.35, rho=0.02, eta=0.50, gamma=0.25, phi=1.5
        )

        # Theta (risk tolerance) should be between 0 and 1
        assert 0 <= prefs.theta <= 1
        # Rho (impatience) typically 1-6%
        assert 0.01 <= prefs.rho <= 0.06
        # Eta (EOIS) should be between 0 and 1
        assert 0 <= prefs.eta <= 1
        # Gamma (intergenerational elasticity) between 0 and 1
        assert 0 <= prefs.gamma <= 1
        # Phi (bequest strength) typically 0-5%
        assert 0 <= prefs.phi <= 5

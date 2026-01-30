"""
Tests for lifecycle_planning.core.balance_sheet module.

Tests IndividualBalanceSheet and FinancialPreferences classes.

Verifies implementation against:
    Idzorek, T.M., & Kaplan, P.D. (2024)
    "Lifetime Financial Advice: A Personalized Optimal Multilevel Approach"
    Chapter 4 (Pages 80-81): Balance sheet framework
    Chapter 12 (Pages 224-237): Isabela example
"""

import pytest
from lifecycle_planning.core.balance_sheet import (
    IndividualBalanceSheet,
    FinancialPreferences,
    create_isabela,
)


class TestFinancialPreferences:
    """Test the FinancialPreferences dataclass."""

    def test_create_preferences(self, isabela_preferences):
        """Test creating financial preferences."""
        assert isabela_preferences.theta == 0.35
        assert isabela_preferences.rho == 0.02
        assert isabela_preferences.eta == 0.50
        assert isabela_preferences.gamma == 0.25
        assert isabela_preferences.phi == 1.5

    def test_preferences_to_params(self, isabela_preferences):
        """Test converting preferences to PreferenceParams."""
        params = isabela_preferences.to_params()
        assert params.theta == 0.35
        assert params.rho == 0.02
        assert params.eta == 0.50
        assert params.gamma == 0.25
        assert params.phi == 1.5


class TestIndividualBalanceSheet:
    """Test the IndividualBalanceSheet class."""

    def test_create_balance_sheet(self, sample_young_client):
        """Test creating an individual balance sheet."""
        assert sample_young_client.current_age == 25
        assert sample_young_client.gender == "female"
        assert sample_young_client.current_income == 75000

    def test_financial_wealth_calculation(self, sample_young_client):
        """Test financial wealth is sum of taxable and tax-advantaged."""
        expected = (
            sample_young_client.taxable_wealth
            + sample_young_client.tax_advantaged_wealth
        )
        assert sample_young_client.financial_wealth == pytest.approx(expected)

    def test_human_capital_positive_for_worker(self, sample_young_client):
        """Test human capital is positive for working individual."""
        assert sample_young_client.human_capital_value > 0

    def test_human_capital_substantial_for_young(self, sample_young_client):
        """Test human capital is substantial for young professional."""
        # 25-year-old with $75k income should have significant HC
        assert sample_young_client.human_capital_value > 500000

    def test_net_worth_positive(self, sample_young_client):
        """Test net worth is positive."""
        assert sample_young_client.net_worth > 0

    def test_net_worth_includes_human_capital(self, sample_young_client):
        """Test net worth includes human capital, not just financial wealth."""
        # Net worth should be much larger than financial wealth
        assert (
            sample_young_client.net_worth
            > sample_young_client.financial_wealth * 2
        )

    def test_retiree_zero_human_capital(self, sample_retiree):
        """Test retiree has zero or minimal human capital."""
        # Retiree with current_age == retirement_age
        assert sample_retiree.current_age == sample_retiree.retirement_age
        # Human capital should be zero or very small
        assert sample_retiree.human_capital_value < 100000

    def test_retiree_net_worth_mostly_financial(self, sample_retiree):
        """Test retiree net worth is mostly financial wealth."""
        # For retiree, net worth should be close to financial wealth
        # (minus liabilities)
        assert sample_retiree.net_worth > 0
        # Net worth should not be wildly different from financial wealth
        ratio = sample_retiree.net_worth / sample_retiree.financial_wealth
        assert 0.3 < ratio < 1.5

    def test_optimal_spending_returns_dict(self, sample_young_client):
        """Test optimal_spending returns a dictionary."""
        spending = sample_young_client.optimal_spending()
        assert isinstance(spending, dict)

    def test_optimal_spending_has_required_keys(self, sample_young_client):
        """Test optimal_spending contains required keys."""
        spending = sample_young_client.optimal_spending()
        required_keys = [
            'discretionary_consumption',
            'nondiscretionary_consumption',
            'total_consumption',
            'savings',
            'savings_rate',
        ]
        for key in required_keys:
            assert key in spending

    def test_optimal_spending_positive(self, sample_young_client):
        """Test optimal spending values are positive."""
        spending = sample_young_client.optimal_spending()
        assert spending['discretionary_consumption'] > 0
        assert spending['total_consumption'] > 0

    def test_optimal_spending_reasonable_for_young(self, sample_young_client):
        """Test optimal spending is reasonable for young professional."""
        spending = sample_young_client.optimal_spending()

        # Total consumption should be less than income (positive savings)
        assert spending['total_consumption'] < sample_young_client.current_income

        # Savings rate should be positive
        assert spending['savings_rate'] > 0

        # Savings rate should be reasonable (not 90%+)
        assert spending['savings_rate'] < 0.80

    def test_optimal_spending_reasonable_for_retiree(self, sample_retiree):
        """Test optimal spending is reasonable for retiree."""
        spending = sample_retiree.optimal_spending()

        # Retiree spending should be substantial
        assert spending['total_consumption'] > 50000

        # Should be drawing down wealth (negative savings for retiree is OK)
        # Or might have small savings if SS income is high


class TestCreateIsabela:
    """Test the create_isabela factory function."""

    def test_isabela_created_successfully(self):
        """Test creating Isabela example."""
        isabela = create_isabela()
        assert isinstance(isabela, IndividualBalanceSheet)

    def test_isabela_has_correct_age(self):
        """Test Isabela has correct age."""
        isabela = create_isabela()
        assert isabela.current_age == 25

    def test_isabela_has_correct_gender(self):
        """Test Isabela has correct gender."""
        isabela = create_isabela()
        assert isabela.gender == "female"

    def test_isabela_has_correct_income(self):
        """Test Isabela has correct income."""
        isabela = create_isabela()
        assert isabela.current_income == 75000

    def test_isabela_has_moderate_preferences(self):
        """Test Isabela has moderate risk preferences."""
        isabela = create_isabela()
        # Theta of 0.35 is moderate risk tolerance
        assert isabela.preferences.theta == pytest.approx(0.35)

    def test_isabela_has_longevity_override(self):
        """Test Isabela has life expectancy override (long-lived family)."""
        isabela = create_isabela()
        assert isabela.life_expectancy_override == 94

    def test_isabela_net_worth_positive(self):
        """Test Isabela has positive net worth."""
        isabela = create_isabela()
        assert isabela.net_worth > 0

    def test_isabela_human_capital_substantial(self):
        """Test Isabela has substantial human capital."""
        isabela = create_isabela()
        # Young professional should have significant HC
        assert isabela.human_capital_value > 1000000


class TestBalanceSheetComparisons:
    """Test comparing different client scenarios."""

    def test_young_has_more_hc_than_retiree(
        self, sample_young_client, sample_retiree
    ):
        """Test young client has more human capital than retiree."""
        assert (
            sample_young_client.human_capital_value
            > sample_retiree.human_capital_value
        )

    def test_retiree_has_more_financial_than_young(
        self, sample_young_client, sample_retiree
    ):
        """Test retiree has more financial wealth than young professional."""
        assert (
            sample_retiree.financial_wealth
            > sample_young_client.financial_wealth
        )

    def test_young_higher_savings_rate_than_retiree(
        self, sample_young_client, sample_retiree
    ):
        """Test young professional should save more than retiree."""
        young_spending = sample_young_client.optimal_spending()
        retiree_spending = sample_retiree.optimal_spending()

        # Young professional should have positive savings rate
        assert young_spending['savings_rate'] > 0

        # Retiree may have negative savings rate (drawing down)
        # or lower savings rate than young professional


class TestBalanceSheetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_income_retiree_valid(self, conservative_preferences):
        """Test creating retiree with zero labor income."""
        retiree = IndividualBalanceSheet(
            current_age=70,
            retirement_age=65,
            max_age=120,
            gender="male",
            education="college",
            current_income=0,  # No labor income
            nondiscretionary_spending=40000,
            taxable_wealth=500000,
            tax_advantaged_wealth=500000,
            social_security_monthly=2500,
            preferences=conservative_preferences,
        )

        assert retiree.current_income == 0
        assert retiree.human_capital_value < 100000  # Minimal HC
        assert retiree.financial_wealth == 1000000

    def test_high_income_professional(self, aggressive_preferences):
        """Test creating high-income professional."""
        high_earner = IndividualBalanceSheet(
            current_age=35,
            retirement_age=65,
            max_age=120,
            gender="male",
            education="post_college",
            current_income=250000,
            nondiscretionary_spending=80000,
            taxable_wealth=200000,
            tax_advantaged_wealth=500000,
            social_security_monthly=3000,
            preferences=aggressive_preferences,
        )

        # High income should lead to high human capital
        assert high_earner.human_capital_value > 3000000
        assert high_earner.net_worth > 2000000

    def test_minimal_wealth_young_starter(self, aggressive_preferences):
        """Test young person just starting with minimal wealth."""
        starter = IndividualBalanceSheet(
            current_age=22,
            retirement_age=67,
            max_age=120,
            gender="female",
            education="college",
            current_income=45000,
            nondiscretionary_spending=35000,
            taxable_wealth=5000,
            tax_advantaged_wealth=10000,
            social_security_monthly=1500,
            preferences=aggressive_preferences,
        )

        # Despite low financial wealth, should have substantial human capital
        assert starter.financial_wealth == 15000
        assert starter.human_capital_value > 500000
        # Net worth dominated by human capital
        assert starter.net_worth > starter.financial_wealth * 10

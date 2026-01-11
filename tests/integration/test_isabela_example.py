"""
Integration test for Isabela case study.

This test verifies that the Isabela example works end-to-end
and produces expected results.
"""

import pytest
import numpy as np
from lifecycle_planning.core.balance_sheet import create_isabela
from lifecycle_planning.core.mortality import (
    life_expectancy,
    survival_probability,
)
from lifecycle_planning.core.portfolio import mvo_optimal_portfolio
from lifecycle_planning.core.annuities import mortality_credit, spia_price


class TestIsabelaExample:
    """Integration tests for the Isabela case study."""

    @pytest.fixture
    def isabela(self):
        """Create Isabela instance for testing."""
        return create_isabela()

    def test_isabela_creation(self, isabela):
        """Test that Isabela instance is created successfully."""
        assert isabela is not None
        assert isabela.current_age == 25
        assert isabela.gender == "female"

    def test_isabela_profile_details(self, isabela):
        """Test Isabela's profile matches expected values."""
        # Age and demographics
        assert isabela.current_age == 25
        assert isabela.retirement_age == 65
        assert isabela.gender == "female"
        assert isabela.education == "post_college"

        # Financial situation
        assert isabela.current_income == 75000
        assert isabela.nondiscretionary_spending == 40000
        assert isabela.tax_advantaged_wealth == 100000
        assert isabela.taxable_wealth == 0

        # Life expectancy override for long-lived family
        assert isabela.life_expectancy_override == 94

    def test_isabela_preferences(self, isabela):
        """Test Isabela's financial preferences."""
        prefs = isabela.preferences

        # Moderate risk tolerance (not aggressive, not conservative)
        assert prefs.theta == pytest.approx(0.35)

        # Patient (low impatience)
        assert prefs.rho == pytest.approx(0.02)

        # Moderate preference for smooth consumption
        assert prefs.eta == pytest.approx(0.50)

        # Low intergenerational elasticity
        assert prefs.gamma == pytest.approx(0.25)

        # Moderate bequest motive
        assert prefs.phi == pytest.approx(1.5)

    def test_isabela_balance_sheet_components(self, isabela):
        """Test Isabela's balance sheet has all components."""
        # Financial wealth
        assert isabela.financial_wealth == 100000

        # Human capital should be substantial (40 years of work ahead)
        assert isabela.human_capital_value > 1000000

        # Liabilities should be present
        assert hasattr(isabela, 'liability_value')

        # Net worth should be positive
        assert isabela.net_worth > 0

    def test_isabela_human_capital_realistic(self, isabela):
        """Test Isabela's human capital is in realistic range."""
        # 25-year-old with $75k income, 40 years to retirement
        # Should have $1.5M - $2.5M in human capital
        hc = isabela.human_capital_value
        assert 1000000 < hc < 3000000

    def test_isabela_net_worth_dominated_by_hc(self, isabela):
        """Test Isabela's net worth is dominated by human capital."""
        # Young professional: HC >> financial wealth
        ratio = isabela.human_capital_value / isabela.financial_wealth
        assert ratio > 5  # HC should be at least 5x financial wealth

    def test_isabela_optimal_spending(self, isabela):
        """Test Isabela's optimal spending is reasonable."""
        spending = isabela.optimal_spending()

        # Should have all required fields
        assert 'discretionary_consumption' in spending
        assert 'nondiscretionary_consumption' in spending
        assert 'total_consumption' in spending
        assert 'savings' in spending
        assert 'savings_rate' in spending

        # Nondiscretionary should match input
        assert spending['nondiscretionary_consumption'] == 40000

        # Discretionary should be positive
        assert spending['discretionary_consumption'] > 0

        # Total should be less than income (saving for retirement)
        assert spending['total_consumption'] < isabela.current_income

        # Should have positive savings rate
        assert spending['savings_rate'] > 0
        assert spending['savings_rate'] < 0.60  # Not extreme

    def test_isabela_spending_rate_reasonable(self, isabela):
        """Test Isabela's spending rate is in reasonable range."""
        spending = isabela.optimal_spending()
        total_consumption = spending['total_consumption']

        # Total consumption should be between $40k (minimum) and $75k (income)
        assert 40000 <= total_consumption <= 75000

        # Should be saving a meaningful amount
        savings = isabela.current_income - total_consumption
        assert savings > 10000  # At least $10k/year savings

    def test_isabela_portfolio_optimization(self, isabela):
        """Test portfolio optimization for Isabela."""
        # With moderate risk tolerance (theta=0.35), should have
        # significant equity allocation but not 100%

        # Using typical market assumptions
        returns = np.array([0.07, 0.03])  # Stocks, bonds
        cov_matrix = np.array([
            [0.15**2, 0.15 * 0.05 * 0.2],
            [0.15 * 0.05 * 0.2, 0.05**2]
        ])

        weights, exp_return, std = mvo_optimal_portfolio(
            returns, cov_matrix, isabela.preferences.theta
        )

        # Moderate risk tolerance should give moderate equity allocation
        equity_weight = weights[0]
        assert 0.40 < equity_weight < 0.90

        # Expected return should be between bond and stock returns
        assert 0.03 < exp_return < 0.07

    def test_isabela_longevity_planning(self, isabela):
        """Test Isabela's longevity planning with family history."""
        # With life expectancy override of 94, her planning should
        # reflect longer lifespan than average

        # Standard life expectancy for 25-year-old female
        standard_le = life_expectancy(25, "female")

        # Her override is 94 - 25 = 69 years remaining
        override_le = isabela.life_expectancy_override - isabela.current_age

        # Override should be longer than standard (long-lived family)
        assert override_le > standard_le

    def test_isabela_survival_to_retirement(self, isabela):
        """Test Isabela's survival probability to retirement."""
        # 25-year-old woman should have very high probability
        # of reaching retirement at 65
        prob = survival_probability(25, 65, "female")
        assert prob > 0.90

    def test_isabela_annuity_analysis_at_retirement(self):
        """Test annuity pricing analysis for Isabela at retirement."""
        # At retirement age 65, female
        # SPIA for $10k/year should cost around $135k-$175k

        # Note: This uses standard mortality, not Isabela's personalized
        from lifecycle_planning.core.mortality import survival_probability_series

        survival_probs = survival_probability_series(65, 40, "female")
        price = spia_price(
            age=65,
            real_rate=0.025,
            survival_probabilities=survival_probs,
            annual_payment=10000,
        )

        # Realistic price range for $10k annuity
        assert 120000 < price < 200000

        # Payout rate should be around 5-7%
        payout_rate = 10000 / price
        assert 0.04 < payout_rate < 0.09

    def test_isabela_mortality_credits(self):
        """Test mortality credits for Isabela at different ages."""
        # Mortality credits should increase with age
        mc_65 = mortality_credit(65, "female", 0.025)
        mc_75 = mortality_credit(75, "female", 0.025)
        mc_85 = mortality_credit(85, "female", 0.025)

        assert mc_65 < mc_75 < mc_85

        # At 75, mortality credit should be meaningful (>5%)
        assert mc_75 > 0.05

    def test_isabela_complete_workflow(self, isabela):
        """Test complete workflow: create, analyze, optimize."""
        # 1. Balance sheet analysis
        assert isabela.net_worth > 0
        assert isabela.human_capital_value > isabela.financial_wealth

        # 2. Spending optimization
        spending = isabela.optimal_spending()
        assert spending['savings_rate'] > 0

        # 3. Portfolio allocation (simulated)
        returns = np.array([0.07, 0.03])
        cov_matrix = np.array([
            [0.15**2, 0.15 * 0.05 * 0.2],
            [0.15 * 0.05 * 0.2, 0.05**2]
        ])
        weights, _, _ = mvo_optimal_portfolio(
            returns, cov_matrix, isabela.preferences.theta
        )
        assert 0 < weights[0] < 1  # Equity weight
        assert 0 < weights[1] < 1  # Bond weight
        assert abs(sum(weights) - 1.0) < 1e-6  # Weights sum to 1

        # All components work together successfully


class TestIsabelaScenarios:
    """Test Isabela's financial picture under different scenarios."""

    @pytest.fixture
    def isabela(self):
        """Create Isabela instance for testing."""
        return create_isabela()

    def test_isabela_wealth_accumulation_trajectory(self, isabela):
        """Test that Isabela is on track for wealth accumulation."""
        # At age 25 with $100k saved and $75k income
        # Should have positive net worth and growing

        initial_wealth = isabela.financial_wealth
        initial_nw = isabela.net_worth

        # Net worth should be much larger than financial wealth
        # due to human capital
        assert initial_nw > initial_wealth * 5

    def test_isabela_income_exceeds_spending(self, isabela):
        """Test Isabela's income exceeds her spending (building wealth)."""
        spending = isabela.optimal_spending()

        # Should be saving (income > total consumption)
        assert isabela.current_income > spending['total_consumption']

        # Savings should be meaningful
        annual_savings = spending['savings']
        assert annual_savings > 15000

    def test_isabela_discretionary_spending_headroom(self, isabela):
        """Test Isabela has room for discretionary spending."""
        spending = isabela.optimal_spending()

        # Discretionary consumption should be positive
        # (above nondiscretionary minimum)
        assert spending['discretionary_consumption'] > 0

        # But not so high that she's not saving
        assert spending['discretionary_consumption'] < 30000

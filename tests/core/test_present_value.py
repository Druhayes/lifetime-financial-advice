"""
Tests for lifecycle_planning.core.present_value module.

Tests present value calculations, human capital, and liability valuations.
"""

import pytest
import numpy as np
from lifecycle_planning.core.present_value import (
    present_value,
    present_value_growing,
    human_capital,
    human_capital_mortality_weighted,
    liability_value,
    liability_value_mortality_weighted,
    net_worth,
    income_discount_rate,
    liability_discount_rate,
)
from lifecycle_planning.core.mortality import survival_probability_series


class TestPresentValue:
    """Test basic present value calculations."""

    def test_pv_single_payment(self):
        """Test PV of single future payment."""
        # $100 in 1 year at 5% discount rate
        pv = present_value(100, 0.05, 1)
        expected = 100 / 1.05
        assert pv == pytest.approx(expected, rel=1e-6)

    def test_pv_multiple_years(self):
        """Test PV of payment multiple years out."""
        # $100 in 10 years at 3% discount rate
        pv = present_value(100, 0.03, 10)
        expected = 100 / (1.03 ** 10)
        assert pv == pytest.approx(expected, rel=1e-6)

    def test_pv_decreases_with_time(self):
        """PV should decrease as time to payment increases."""
        pv_1yr = present_value(100, 0.05, 1)
        pv_5yr = present_value(100, 0.05, 5)
        pv_10yr = present_value(100, 0.05, 10)

        assert pv_1yr > pv_5yr > pv_10yr

    def test_pv_decreases_with_discount_rate(self):
        """PV should decrease as discount rate increases."""
        pv_2pct = present_value(100, 0.02, 10)
        pv_5pct = present_value(100, 0.05, 10)
        pv_8pct = present_value(100, 0.08, 10)

        assert pv_2pct > pv_5pct > pv_8pct

    def test_pv_zero_discount_rate(self):
        """PV with zero discount rate equals future value."""
        pv = present_value(100, 0.0, 10)
        assert pv == pytest.approx(100)

    def test_pv_zero_years(self):
        """PV of immediate payment equals payment amount."""
        pv = present_value(100, 0.05, 0)
        assert pv == pytest.approx(100)


class TestPresentValueGrowing:
    """Test present value of growing annuity."""

    def test_pv_growing_annuity(self):
        """Test PV of growing annuity stream."""
        # $50k/year for 30 years, growing at 2%, discounted at 5%
        pv = present_value_growing(50000, 0.05, 0.02, 30)

        # Should be substantial (around $1M+)
        assert 800000 < pv < 1500000

    def test_pv_growing_increases_with_growth_rate(self):
        """PV should increase as growth rate increases."""
        pv_0pct = present_value_growing(50000, 0.05, 0.00, 20)
        pv_2pct = present_value_growing(50000, 0.05, 0.02, 20)
        pv_3pct = present_value_growing(50000, 0.05, 0.03, 20)

        assert pv_0pct < pv_2pct < pv_3pct

    def test_pv_growing_zero_growth_matches_annuity(self):
        """PV with zero growth should match standard annuity formula."""
        payment = 10000
        rate = 0.04
        years = 10

        pv_growing = present_value_growing(payment, rate, 0.0, years)

        # Standard annuity formula
        pv_annuity = payment * (1 - (1 + rate) ** -years) / rate

        assert pv_growing == pytest.approx(pv_annuity, rel=1e-6)

    def test_pv_growing_one_payment(self):
        """PV of single growing payment."""
        pv = present_value_growing(50000, 0.05, 0.02, 1)
        # First payment is immediate (no growth yet)
        expected = 50000 / 1.05
        assert pv == pytest.approx(expected, rel=1e-3)


class TestHumanCapital:
    """Test human capital calculations."""

    def test_human_capital_positive(self):
        """Human capital should be positive for working individual."""
        hc = human_capital(
            income=75000,
            years_to_retirement=40,
            discount_rate=0.03,
            income_growth=0.02,
        )
        assert hc > 0

    def test_human_capital_realistic_value(self):
        """Human capital should be in realistic range."""
        # $75k income for 40 years should be $1.5M - $2.5M in PV
        hc = human_capital(
            income=75000,
            years_to_retirement=40,
            discount_rate=0.03,
            income_growth=0.02,
        )
        assert 1000000 < hc < 3000000

    def test_human_capital_increases_with_income(self):
        """Human capital should increase with higher income."""
        hc_50k = human_capital(50000, 30, 0.03, 0.02)
        hc_100k = human_capital(100000, 30, 0.03, 0.02)

        assert hc_100k == pytest.approx(2 * hc_50k, rel=0.01)

    def test_human_capital_decreases_with_shorter_horizon(self):
        """Human capital should decrease with fewer working years."""
        hc_40yr = human_capital(75000, 40, 0.03, 0.02)
        hc_20yr = human_capital(75000, 20, 0.03, 0.02)
        hc_10yr = human_capital(75000, 10, 0.03, 0.02)

        assert hc_40yr > hc_20yr > hc_10yr

    def test_human_capital_zero_years(self):
        """Human capital with zero working years should be zero."""
        hc = human_capital(75000, 0, 0.03, 0.02)
        assert hc == pytest.approx(0)


class TestHumanCapitalMortalityWeighted:
    """Test mortality-weighted human capital."""

    def test_mortality_weighted_hc_positive(self):
        """Mortality-weighted HC should be positive."""
        survival_probs = survival_probability_series(25, 40, "female")
        hc = human_capital_mortality_weighted(
            income=75000,
            discount_rate=0.03,
            income_growth=0.02,
            survival_probabilities=survival_probs,
        )
        assert hc > 0

    def test_mortality_weighted_lower_than_certain(self):
        """Mortality-weighted HC should be lower than certain HC."""
        survival_probs = survival_probability_series(65, 20, "male")

        hc_mortality = human_capital_mortality_weighted(
            income=50000,
            discount_rate=0.03,
            income_growth=0.02,
            survival_probabilities=survival_probs,
        )

        hc_certain = human_capital(
            income=50000,
            years_to_retirement=20,
            discount_rate=0.03,
            income_growth=0.02,
        )

        # Mortality risk should reduce HC
        assert hc_mortality < hc_certain

    def test_mortality_weighted_young_vs_old(self):
        """Young person should have HC closer to certain HC than old person."""
        # Young person (25) vs older person (55)
        survival_young = survival_probability_series(25, 30, "female")
        survival_old = survival_probability_series(55, 10, "male")

        hc_young = human_capital_mortality_weighted(
            income=75000, discount_rate=0.03, income_growth=0.02,
            survival_probabilities=survival_young
        )
        hc_old = human_capital_mortality_weighted(
            income=75000, discount_rate=0.03, income_growth=0.02,
            survival_probabilities=survival_old
        )

        # Young person has higher survival, so higher HC
        assert hc_young > hc_old


class TestLiabilityValue:
    """Test liability value calculations."""

    def test_liability_value_positive(self):
        """Liability value should be positive."""
        lv = liability_value(
            annual_spending=40000,
            years=40,
            discount_rate=0.025,
            spending_growth=0.015,
        )
        assert lv > 0

    def test_liability_value_realistic(self):
        """Liability value should be in realistic range."""
        # $40k spending for 40 years should be $800k - $1.5M in PV
        lv = liability_value(40000, 40, 0.025, 0.015)
        assert 600000 < lv < 1800000

    def test_liability_increases_with_spending(self):
        """Liability should increase with higher spending."""
        lv_30k = liability_value(30000, 30, 0.025, 0.015)
        lv_60k = liability_value(60000, 30, 0.025, 0.015)

        assert lv_60k == pytest.approx(2 * lv_30k, rel=0.01)


class TestLiabilityValueMortalityWeighted:
    """Test mortality-weighted liability calculations."""

    def test_mortality_weighted_liability_positive(self):
        """Mortality-weighted liability should be positive."""
        survival_probs = survival_probability_series(65, 30, "female")
        lv = liability_value_mortality_weighted(
            annual_spending=50000,
            discount_rate=0.025,
            spending_growth=0.015,
            survival_probabilities=survival_probs,
        )
        assert lv > 0

    def test_mortality_weighted_lower_than_certain(self):
        """Mortality-weighted liability should be lower than certain."""
        survival_probs = survival_probability_series(70, 25, "male")

        lv_mortality = liability_value_mortality_weighted(
            annual_spending=50000,
            discount_rate=0.025,
            spending_growth=0.015,
            survival_probabilities=survival_probs,
        )

        lv_certain = liability_value(
            annual_spending=50000,
            years=25,
            discount_rate=0.025,
            spending_growth=0.015,
        )

        # Mortality risk should reduce liability
        assert lv_mortality < lv_certain


class TestNetWorth:
    """Test net worth calculations."""

    def test_net_worth_basic(self):
        """Test basic net worth calculation."""
        nw = net_worth(
            financial_wealth=200000,
            human_capital=1500000,
            liabilities=800000,
        )
        expected = 200000 + 1500000 - 800000
        assert nw == pytest.approx(expected)

    def test_net_worth_can_be_negative(self):
        """Net worth can be negative if liabilities exceed assets."""
        nw = net_worth(
            financial_wealth=50000,
            human_capital=300000,
            liabilities=500000,
        )
        assert nw < 0
        assert nw == pytest.approx(-150000)

    def test_net_worth_zero_human_capital(self):
        """Net worth for retiree with no human capital."""
        nw = net_worth(
            financial_wealth=1000000,
            human_capital=0,
            liabilities=600000,
        )
        assert nw == pytest.approx(400000)


class TestDiscountRates:
    """Test income and liability discount rate calculations."""

    def test_income_discount_rate_positive(self):
        """Income discount rate should be positive."""
        dr = income_discount_rate(
            risk_free_rate=0.025,
            equity_return=0.07,
            hc_equity_weight=0.20,
        )
        assert dr > 0

    def test_income_discount_rate_above_risk_free(self):
        """Income discount rate should be above risk-free rate."""
        rf = 0.025
        dr = income_discount_rate(
            risk_free_rate=rf,
            equity_return=0.07,
            hc_equity_weight=0.20,
        )
        assert dr > rf

    def test_income_discount_rate_below_equity(self):
        """Income discount rate should be below equity return."""
        equity = 0.07
        dr = income_discount_rate(
            risk_free_rate=0.025,
            equity_return=equity,
            hc_equity_weight=0.20,
        )
        assert dr < equity

    def test_income_discount_rate_increases_with_equity_weight(self):
        """Higher equity weight should increase discount rate."""
        dr_20 = income_discount_rate(0.025, 0.07, 0.20)
        dr_40 = income_discount_rate(0.025, 0.07, 0.40)
        dr_60 = income_discount_rate(0.025, 0.07, 0.60)

        assert dr_20 < dr_40 < dr_60

    def test_liability_discount_rate_positive(self):
        """Liability discount rate should be positive."""
        dr = liability_discount_rate(
            risk_free_rate=0.025,
            equity_return=0.07,
            liability_equity_weight=0.15,
        )
        assert dr > 0

    def test_liability_discount_rate_lower_than_income(self):
        """Liability discount rate should typically be lower than income."""
        # Liabilities usually have lower equity weight than human capital
        dr_income = income_discount_rate(0.025, 0.07, 0.20)
        dr_liability = liability_discount_rate(0.025, 0.07, 0.10)

        assert dr_liability < dr_income

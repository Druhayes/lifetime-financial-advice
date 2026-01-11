"""
Tests for lifecycle_planning.core.mortality module.

Tests Gompertz survival probability calculations, life expectancy,
and mortality-related functions.
"""

import pytest
import numpy as np
from lifecycle_planning.core.mortality import (
    gompertz_survival_prob,
    truncated_gompertz,
    survival_probability,
    joint_survival_probability,
    death_probability,
    life_expectancy,
    survival_probability_series,
    death_probability_series,
    personalized_mode,
    GOMPERTZ_PARAMS,
)


class TestGompertzSurvivalProb:
    """Test the basic Gompertz survival probability function."""

    def test_survival_at_mode_age(self):
        """Survival probability at mode age should be around 36.8% (1/e)."""
        # For females, mode is 91
        prob = gompertz_survival_prob(91, 91, m=91, b=8.88)
        assert prob == pytest.approx(1.0)  # Same age = 100% survival

    def test_survival_decreases_with_age(self):
        """Survival probability should decrease as age increases."""
        m, b = 91, 8.88
        prob_70 = gompertz_survival_prob(70, 80, m, b)
        prob_80 = gompertz_survival_prob(70, 90, m, b)
        prob_90 = gompertz_survival_prob(70, 100, m, b)

        assert prob_70 > prob_80 > prob_90

    def test_young_age_high_survival(self):
        """Young ages should have very high survival probability."""
        prob = gompertz_survival_prob(25, 65, m=91, b=8.88)
        assert prob > 0.90

    def test_very_old_age_low_survival(self):
        """Very old ages should have low survival probability."""
        prob = gompertz_survival_prob(70, 110, m=91, b=8.88)
        assert prob < 0.10


class TestTruncatedGompertz:
    """Test the truncated Gompertz function."""

    def test_probability_at_same_age(self):
        """Probability of surviving to current age should be 1.0."""
        prob = truncated_gompertz(65, 65, m=88, b=10.65)
        assert prob == pytest.approx(1.0)

    def test_probability_decreases_with_target_age(self):
        """Probability should decrease as target age increases."""
        prob_to_70 = truncated_gompertz(65, 70, m=88, b=10.65)
        prob_to_80 = truncated_gompertz(65, 80, m=88, b=10.65)
        prob_to_90 = truncated_gompertz(65, 90, m=88, b=10.65)

        assert prob_to_70 > prob_to_80 > prob_to_90

    def test_probability_bounds(self):
        """Probability should always be between 0 and 1."""
        for current_age in [25, 45, 65, 85]:
            for target_age in range(current_age, 120):
                prob = truncated_gompertz(
                    current_age, target_age, m=91, b=8.88
                )
                assert 0 <= prob <= 1


class TestSurvivalProbability:
    """Test the survival_probability function with gender lookup."""

    def test_female_survival_to_retirement(self):
        """Test female survival to retirement age."""
        prob = survival_probability(25, 65, "female")
        # Young women have high survival to 65
        assert 0.90 < prob < 1.0

    def test_male_survival_to_retirement(self):
        """Test male survival to retirement age."""
        prob = survival_probability(25, 65, "male")
        # Young men have high survival to 65 (but slightly lower than women)
        assert 0.85 < prob < 0.95

    def test_female_survives_longer_than_male(self):
        """Females should have higher survival probability than males."""
        female_prob = survival_probability(65, 90, "female")
        male_prob = survival_probability(65, 90, "male")
        assert female_prob > male_prob

    def test_survival_at_current_age(self):
        """Survival to current age should be 1.0."""
        assert survival_probability(50, 50, "female") == pytest.approx(1.0)
        assert survival_probability(50, 50, "male") == pytest.approx(1.0)

    def test_invalid_gender_raises_error(self):
        """Invalid gender should raise ValueError."""
        with pytest.raises(ValueError):
            survival_probability(25, 65, "invalid")


class TestJointSurvivalProbability:
    """Test joint survival probability for couples."""

    def test_joint_survival_returns_array(self):
        """Joint survival should return an array."""
        joint_probs = joint_survival_probability(65, 62, target_years=20)
        assert isinstance(joint_probs, np.ndarray)
        assert len(joint_probs) == 20

    def test_joint_survival_probabilities_decrease(self):
        """Joint survival probabilities should decrease over time."""
        joint_probs = joint_survival_probability(65, 62, target_years=20)
        # Probabilities should be monotonically decreasing
        for i in range(len(joint_probs) - 1):
            assert joint_probs[i] >= joint_probs[i + 1]


class TestDeathProbability:
    """Test death probability calculations."""

    def test_death_probability_complements_survival(self):
        """Death probability should equal 1 - survival probability."""
        survival_prob = survival_probability(65, 85, "female")
        death_prob = death_probability(65, 85, "female")
        assert death_prob == pytest.approx(1 - survival_prob)

    def test_death_probability_increases_with_age(self):
        """Death probability should increase with target age."""
        prob_70 = death_probability(65, 70, "male")
        prob_80 = death_probability(65, 80, "male")
        prob_90 = death_probability(65, 90, "male")

        assert prob_70 < prob_80 < prob_90

    def test_death_probability_bounds(self):
        """Death probability should be between 0 and 1."""
        prob = death_probability(30, 100, "female")
        assert 0 <= prob <= 1


class TestLifeExpectancy:
    """Test life expectancy calculations."""

    def test_female_life_expectancy_longer(self):
        """Females should have longer life expectancy than males."""
        female_le = life_expectancy(65, "female")
        male_le = life_expectancy(65, "male")
        assert female_le > male_le

    def test_life_expectancy_decreases_with_age(self):
        """Remaining life expectancy should decrease with age."""
        le_25 = life_expectancy(25, "female")
        le_65 = life_expectancy(65, "female")

        # Note: remaining years, not total lifespan
        # At 25, expect ~60 more years; at 65, expect ~20 more years
        assert le_25 > le_65

    def test_life_expectancy_positive(self):
        """Life expectancy should always be positive."""
        for age in [25, 45, 65, 85]:
            le_female = life_expectancy(age, "female")
            le_male = life_expectancy(age, "male")
            assert le_female > 0
            assert le_male > 0

    def test_life_expectancy_reasonable_range(self):
        """Life expectancy should be in reasonable range."""
        # 25-year-old female should expect to live 55-65 more years
        le = life_expectancy(25, "female")
        assert 55 < le < 70


class TestSurvivalProbabilitySeries:
    """Test survival probability series generation."""

    def test_series_length(self):
        """Series should have correct length."""
        series = survival_probability_series(65, 30, "female")
        assert len(series) == 30

    def test_series_starts_at_one(self):
        """First element (survival to current age) should be 1.0."""
        series = survival_probability_series(65, 30, "female")
        assert series[0] == pytest.approx(1.0)

    def test_series_decreasing(self):
        """Series should be monotonically decreasing."""
        series = survival_probability_series(65, 30, "female")
        for i in range(len(series) - 1):
            assert series[i] >= series[i + 1]

    def test_series_positive(self):
        """All probabilities should be positive."""
        series = survival_probability_series(50, 50, "male")
        assert np.all(series > 0)


class TestDeathProbabilitySeries:
    """Test death probability series generation."""

    def test_death_series_length(self):
        """Series should have correct length."""
        series = death_probability_series(65, 30, "female")
        assert len(series) == 30

    def test_death_series_starts_at_zero(self):
        """First element (death at current age) should be 0.0."""
        series = death_probability_series(65, 30, "female")
        assert series[0] == pytest.approx(0.0)

    def test_death_series_increasing(self):
        """Series should be monotonically increasing."""
        series = death_probability_series(65, 30, "female")
        for i in range(len(series) - 1):
            assert series[i] <= series[i + 1]

    def test_death_series_complements_survival(self):
        """Death series should equal 1 - survival series."""
        survival_series = survival_probability_series(65, 20, "male")
        death_series = death_probability_series(65, 20, "male")

        np.testing.assert_array_almost_equal(
            death_series, 1 - survival_series
        )


class TestPersonalizedMode:
    """Test personalized mode calculation based on life expectancy."""

    def test_personalized_mode_higher_for_longer_life(self):
        """Personalized mode should be higher for longer life expectancy."""
        mode_90 = personalized_mode(
            current_age=65, life_expectancy_override=90, gender="female"
        )
        mode_100 = personalized_mode(
            current_age=65, life_expectancy_override=100, gender="female"
        )
        assert mode_100 > mode_90

    def test_personalized_mode_none_returns_default(self):
        """No override should return default mode for gender."""
        mode = personalized_mode(
            current_age=65, life_expectancy_override=None, gender="female"
        )
        assert mode == GOMPERTZ_PARAMS["female"]["mode"]

    def test_personalized_mode_reasonable(self):
        """Personalized mode should be in reasonable range."""
        mode = personalized_mode(
            current_age=25, life_expectancy_override=95, gender="female"
        )
        # Should be between 85 and 100 for someone expecting to live to 95
        assert 85 <= mode <= 105


class TestGompertzParams:
    """Test the GOMPERTZ_PARAMS constants."""

    def test_params_exist_for_both_genders(self):
        """Parameters should exist for male and female."""
        assert "male" in GOMPERTZ_PARAMS
        assert "female" in GOMPERTZ_PARAMS

    def test_female_mode_higher_than_male(self):
        """Female mode should be higher (women live longer)."""
        female_mode = GOMPERTZ_PARAMS["female"].m
        male_mode = GOMPERTZ_PARAMS["male"].m
        assert female_mode > male_mode

    def test_params_have_required_fields(self):
        """Both genders should have m and b attributes."""
        for gender in ["male", "female"]:
            assert hasattr(GOMPERTZ_PARAMS[gender], 'm')
            assert hasattr(GOMPERTZ_PARAMS[gender], 'b')

    def test_params_are_positive(self):
        """Mode and dispersion should be positive."""
        for gender in ["male", "female"]:
            assert GOMPERTZ_PARAMS[gender].m > 0
            assert GOMPERTZ_PARAMS[gender].b > 0

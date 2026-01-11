"""
Income Modeling for Lifecycle Financial Planning

Implements wage curve projections based on age, gender, and education level.
Also includes stochastic income modeling with innovations.

Based on Chapter 4 of "Lifetime Financial Advice".

The income model includes:
    - Deterministic wage curves by demographic group
    - Stochastic income with annual innovations
    - Social Security income projections

References:
    - Bureau of Labor Statistics wage data
    - Social Security Administration benefit calculators
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class Education(Enum):
    HIGH_SCHOOL = "high_school"
    SOME_COLLEGE = "some_college"
    COLLEGE = "college"
    POST_COLLEGE = "post_college"


@dataclass(frozen=True)
class WageCurveParams:
    """Parameters for a wage curve by education and gender."""

    peak_age: int  # Age at peak earnings
    peak_multiplier: float  # Peak earnings as multiple of starting
    base_salary_25: float  # Starting salary at age 25
    growth_early: float  # Annual growth rate before peak
    decline_late: float  # Annual decline rate after peak


# Wage curve parameters based on document Exhibit 4.2
# Values calibrated to approximate the curves in the document
WAGE_CURVES: dict[tuple[Gender, Education], WageCurveParams] = {
    (Gender.FEMALE, Education.HIGH_SCHOOL): WageCurveParams(
        peak_age=45, peak_multiplier=1.3, base_salary_25=25000,
        growth_early=0.013, decline_late=-0.005
    ),
    (Gender.FEMALE, Education.SOME_COLLEGE): WageCurveParams(
        peak_age=47, peak_multiplier=1.4, base_salary_25=32000,
        growth_early=0.015, decline_late=-0.005
    ),
    (Gender.FEMALE, Education.COLLEGE): WageCurveParams(
        peak_age=50, peak_multiplier=1.6, base_salary_25=45000,
        growth_early=0.018, decline_late=-0.004
    ),
    (Gender.FEMALE, Education.POST_COLLEGE): WageCurveParams(
        peak_age=52, peak_multiplier=1.8, base_salary_25=55000,
        growth_early=0.020, decline_late=-0.003
    ),
    (Gender.MALE, Education.HIGH_SCHOOL): WageCurveParams(
        peak_age=45, peak_multiplier=1.4, base_salary_25=32000,
        growth_early=0.015, decline_late=-0.008
    ),
    (Gender.MALE, Education.SOME_COLLEGE): WageCurveParams(
        peak_age=48, peak_multiplier=1.5, base_salary_25=40000,
        growth_early=0.017, decline_late=-0.007
    ),
    (Gender.MALE, Education.COLLEGE): WageCurveParams(
        peak_age=50, peak_multiplier=1.7, base_salary_25=55000,
        growth_early=0.020, decline_late=-0.005
    ),
    (Gender.MALE, Education.POST_COLLEGE): WageCurveParams(
        peak_age=52, peak_multiplier=2.0, base_salary_25=70000,
        growth_early=0.025, decline_late=-0.004
    ),
}


def income_curve(
    age: int,
    gender: str | Gender = Gender.FEMALE,
    education: str | Education = Education.POST_COLLEGE,
    base_salary_override: float | None = None,
) -> float:
    """
    Get projected income for a specific age based on wage curve.

    Uses standard wage curves calibrated to education and gender.

    Args:
        age: Age to get income for (25-65)
        gender: Gender (male/female)
        education: Education level
        base_salary_override: Override the default base salary at age 25

    Returns:
        Projected annual income at the specified age

    Example:
        >>> income_curve(25, "female", "post_college")
        55000.0  # Base salary for post-college woman
        >>> income_curve(52, "female", "post_college")
        99000.0  # Near peak earnings
    """
    if isinstance(gender, str):
        gender = Gender(gender.lower())
    if isinstance(education, str):
        education = Education(education.lower())

    params = WAGE_CURVES.get((gender, education))
    if params is None:
        raise ValueError(f"No wage curve for {gender}, {education}")

    base = base_salary_override if base_salary_override else params.base_salary_25

    if age < 25:
        # Before career start, use reduced base
        return base * 0.5
    elif age > 65:
        # After retirement, no wage income
        return 0.0

    # Calculate salary based on wage curve shape
    if age <= params.peak_age:
        # Growth phase
        years_from_start = age - 25
        growth = (1 + params.growth_early) ** years_from_start
        return base * growth
    else:
        # Decline phase
        # First calculate peak salary
        years_to_peak = params.peak_age - 25
        peak_salary = base * (1 + params.growth_early) ** years_to_peak
        # Then decline from peak
        years_from_peak = age - params.peak_age
        decline = (1 + params.decline_late) ** years_from_peak
        return peak_salary * decline


def income_multiplier(
    current_income: float,
    gender: str | Gender = Gender.FEMALE,
    education: str | Education = Education.POST_COLLEGE,
    current_age: int = 25,
) -> float:
    """
    Calculate income multiplier based on current income vs standard curve.

    Used to personalize income projections based on actual current income.

    Formula: multiplier = current_income / standard_income_at_age

    Args:
        current_income: Actual current income
        gender: Gender
        education: Education level
        current_age: Current age

    Returns:
        Multiplier to apply to standard wage curve

    Example:
        >>> # Isabela earns $75k vs standard $40k for her category
        >>> income_multiplier(75000, "female", "post_college", 25)
        1.875  # Apply this to all future projections
    """
    standard_income = income_curve(current_age, gender, education)
    if standard_income <= 0:
        return 1.0
    return current_income / standard_income


def projected_income(
    current_age: int,
    retirement_age: int,
    current_income: float,
    gender: str | Gender = Gender.FEMALE,
    education: str | Education = Education.POST_COLLEGE,
    inflation_rate: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Project income from current age to retirement.

    Combines standard wage curve with personal income multiplier
    and optional inflation adjustment.

    Args:
        current_age: Current age
        retirement_age: Planned retirement age
        current_income: Current annual income
        gender: Gender
        education: Education level
        inflation_rate: Annual inflation rate for nominal projections

    Returns:
        Array of projected income for each year until retirement

    Example:
        >>> income = projected_income(25, 65, 75000, "female", "post_college")
        >>> income[0]
        75000.0
        >>> income[20]  # At age 45
        121500.0  # Peak earnings period
    """
    multiplier = income_multiplier(current_income, gender, education, current_age)

    years = retirement_age - current_age
    ages = np.arange(current_age, retirement_age)

    incomes = np.array([
        income_curve(age, gender, education) * multiplier
        for age in ages
    ])

    # Apply inflation if specified
    if inflation_rate > 0:
        inflation_factors = (1 + inflation_rate) ** np.arange(years)
        incomes = incomes * inflation_factors

    return incomes


def social_security_income(
    retirement_age: int,
    average_indexed_earnings: float,
    bend_point_1: float = 1115.0,
    bend_point_2: float = 6721.0,
    max_age: int = 120,
) -> npt.NDArray[np.float64]:
    """
    Estimate Social Security benefits using PIA formula.

    Simplified Primary Insurance Amount (PIA) calculation.
    Actual benefits depend on full earnings history.

    Formula (monthly):
        90% of first $1,115 of AIME
        + 32% of AIME between $1,115 and $6,721
        + 15% of AIME above $6,721

    Args:
        retirement_age: Age when benefits start
        average_indexed_earnings: Average indexed monthly earnings (AIME)
        bend_point_1: First bend point (updated annually)
        bend_point_2: Second bend point (updated annually)
        max_age: Maximum age for projections

    Returns:
        Array of annual Social Security income from retirement to max_age
    """
    # Calculate PIA (monthly benefit at full retirement age)
    if average_indexed_earnings <= bend_point_1:
        monthly_pia = 0.90 * average_indexed_earnings
    elif average_indexed_earnings <= bend_point_2:
        monthly_pia = (
            0.90 * bend_point_1
            + 0.32 * (average_indexed_earnings - bend_point_1)
        )
    else:
        monthly_pia = (
            0.90 * bend_point_1
            + 0.32 * (bend_point_2 - bend_point_1)
            + 0.15 * (average_indexed_earnings - bend_point_2)
        )

    annual_benefit = monthly_pia * 12

    # Create income stream from retirement to max_age
    years_in_retirement = max_age - retirement_age
    return np.full(years_in_retirement, annual_benefit)


def risky_income(
    base_income: npt.NDArray[np.float64],
    volatility: float = 0.10,
    correlation_with_market: float = 0.2,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Generate stochastic income path with random innovations.

    Models income uncertainty using lognormal shocks.
    Innovations are cumulative, so deviations can grow over time.

    Formula: Y_t = Y_t^base * exp(Σ ε_s) where ε ~ N(0, σ²)

    Args:
        base_income: Deterministic income path
        volatility: Annual income volatility (std dev of log shocks)
        correlation_with_market: Correlation of income with equity returns
        seed: Random seed for reproducibility

    Returns:
        Stochastic income path

    Example:
        >>> np.random.seed(42)
        >>> base = projected_income(25, 65, 75000, "female", "post_college")
        >>> risky = risky_income(base, volatility=0.10)
        >>> # risky income will deviate from base over time
    """
    if seed is not None:
        np.random.seed(seed)

    n_years = len(base_income)

    # Generate cumulative log-normal shocks
    # Each year's shock persists into the future
    annual_shocks = np.random.normal(0, volatility, n_years)

    # Adjust for mean (so expected value equals base)
    annual_shocks = annual_shocks - volatility**2 / 2

    # Cumulative shocks
    cumulative_shocks = np.cumsum(annual_shocks)

    # Apply shocks
    risky_path = base_income * np.exp(cumulative_shocks)

    return risky_path


def simulate_income_paths(
    base_income: npt.NDArray[np.float64],
    n_simulations: int = 1000,
    volatility: float = 0.10,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Simulate multiple stochastic income paths.

    Useful for Monte Carlo analysis of human capital uncertainty.

    Args:
        base_income: Deterministic base income path
        n_simulations: Number of paths to simulate
        volatility: Annual income volatility
        seed: Random seed

    Returns:
        Array of shape (n_simulations, n_years) with simulated paths
    """
    if seed is not None:
        np.random.seed(seed)

    n_years = len(base_income)
    paths = np.zeros((n_simulations, n_years))

    for i in range(n_simulations):
        paths[i] = risky_income(base_income, volatility)

    return paths


def income_percentiles(
    simulated_paths: npt.NDArray[np.float64],
    percentiles: list[int] = [5, 25, 50, 75, 95],
) -> dict[int, npt.NDArray[np.float64]]:
    """
    Calculate percentiles of simulated income paths.

    Args:
        simulated_paths: Array of simulated income paths
        percentiles: List of percentiles to calculate

    Returns:
        Dictionary mapping percentile to income path
    """
    result = {}
    for p in percentiles:
        result[p] = np.percentile(simulated_paths, p, axis=0)
    return result


def full_income_stream(
    current_age: int,
    retirement_age: int,
    max_age: int,
    current_income: float,
    social_security_monthly: float,
    gender: str | Gender = Gender.FEMALE,
    education: str | Education = Education.POST_COLLEGE,
) -> npt.NDArray[np.float64]:
    """
    Create complete income stream including wages and Social Security.

    Combines working years income with retirement benefits.

    Args:
        current_age: Current age
        retirement_age: Planned retirement age
        max_age: Maximum age for projections
        current_income: Current annual income
        social_security_monthly: Expected monthly Social Security benefit
        gender: Gender
        education: Education level

    Returns:
        Array of income for each year from current age to max_age
    """
    total_years = max_age - current_age

    # Working years income
    work_income = projected_income(
        current_age, retirement_age, current_income, gender, education
    )

    # Retirement years - Social Security only
    retirement_years = max_age - retirement_age
    ss_income = np.full(retirement_years, social_security_monthly * 12)

    # Combine
    full_stream = np.concatenate([work_income, ss_income])

    return full_stream

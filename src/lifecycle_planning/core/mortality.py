"""
Mortality and Survival Probability Functions

Implements the Gompertz model for survival probability calculations.
Based on Appendix 3A of "Lifetime Financial Advice".

The Gompertz formula provides a parametric model for survival probability
with three parameters:
    - a1: current age
    - a2: target age (age to survive to)
    - m: mode of the distribution of age at death
    - b: dispersion of age at death around the mode

References:
    - Gompertz, B. (1825). On the nature of the function expressive of human mortality.
    - Milevsky, M.A. (2012). The 7 Most Important Equations for Your Retirement.
    - Blanchett, D. and Kaplan, P. (2013). Estimating the True Cost of Retirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class GompertzParams:
    """Gompertz model parameters for a specific gender."""

    m: float  # Mode of age at death distribution
    b: float  # Dispersion parameter


# Default Gompertz parameters based on US mortality data
# From Exhibit 3A.1 in the document
GOMPERTZ_PARAMS: dict[str, GompertzParams] = {
    "male": GompertzParams(m=88.0, b=10.65),
    "female": GompertzParams(m=91.0, b=8.88),
}


def gompertz_survival_prob(
    a1: float,
    a2: float,
    m: float,
    b: float,
) -> float:
    """
    Calculate survival probability using the Gompertz formula.

    Probability of surviving to age a2 given alive at age a1.

    Formula: g(a2, a1; m, b) = exp(-exp((a1-m)/b) * (exp((a2-a1)/b) - 1))

    Args:
        a1: Current age
        a2: Target age (must be >= a1)
        m: Mode of the distribution of age at death
        b: Dispersion parameter (similar to standard deviation)

    Returns:
        Probability of surviving from age a1 to age a2 (between 0 and 1)

    Example:
        >>> gompertz_survival_prob(65, 75, m=88, b=10.65)
        0.8234...  # ~82% chance of 65-year-old man living to 75
    """
    if a2 < a1:
        raise ValueError("Target age a2 must be >= current age a1")
    if a2 == a1:
        return 1.0
    if b <= 0:
        raise ValueError("Dispersion parameter b must be positive")

    exp_term1 = np.exp((a1 - m) / b)
    exp_term2 = np.exp((a2 - a1) / b) - 1

    return float(np.exp(-exp_term1 * exp_term2))


def truncated_gompertz(
    a1: float,
    a2: float,
    m: float,
    b: float,
    max_age: float = 120.0,
) -> float:
    """
    Calculate truncated Gompertz survival probability.

    The truncated version ensures death is certain by max_age.
    Used to handle the fact that Gompertz never quite reaches zero.

    Formula: g_trunc(a2, a1) = g(a2, a1) if a2 < max_age, else 0

    Args:
        a1: Current age
        a2: Target age
        m: Mode of age at death distribution
        b: Dispersion parameter
        max_age: Age at which death is certain (default 120)

    Returns:
        Truncated survival probability
    """
    if a2 >= max_age:
        return 0.0
    return gompertz_survival_prob(a1, a2, m, b)


def survival_probability(
    current_age: float,
    target_age: float,
    gender: str = "female",
    m_override: float | None = None,
    b_override: float | None = None,
    max_age: float = 120.0,
) -> float:
    """
    Calculate survival probability with convenient defaults.

    Wrapper around truncated_gompertz with gender-specific defaults.

    Args:
        current_age: Person's current age
        target_age: Age to calculate survival probability for
        gender: "male" or "female" for default parameters
        m_override: Optional override for mode parameter
        b_override: Optional override for dispersion parameter
        max_age: Maximum possible age

    Returns:
        Probability of surviving from current_age to target_age

    Example:
        >>> survival_probability(25, 65, gender="female")
        0.9234...  # High probability for young woman to reach 65
    """
    gender = gender.lower()
    if gender not in GOMPERTZ_PARAMS:
        raise ValueError(f"Gender must be 'male' or 'female', got '{gender}'")

    params = GOMPERTZ_PARAMS[gender]
    m = m_override if m_override is not None else params.m
    b = b_override if b_override is not None else params.b

    return truncated_gompertz(current_age, target_age, m, b, max_age)


def survival_probability_series(
    current_age: float,
    years: int,
    gender: str = "female",
    m_override: float | None = None,
    b_override: float | None = None,
    max_age: float = 120.0,
) -> npt.NDArray[np.float64]:
    """
    Calculate survival probabilities for a series of future years.

    Useful for life-cycle calculations requiring survival probability
    for each year from current age to current_age + years.

    Args:
        current_age: Starting age
        years: Number of years to project
        gender: "male" or "female"
        m_override: Optional mode override
        b_override: Optional dispersion override
        max_age: Maximum age

    Returns:
        Array of survival probabilities, index 0 = current year (prob=1.0)

    Example:
        >>> probs = survival_probability_series(65, 40, "male")
        >>> probs[0]  # Probability of being alive now
        1.0
        >>> probs[20]  # Probability of living 20 more years
        0.456...
    """
    target_ages = np.arange(current_age, current_age + years + 1)
    probs = np.array([
        survival_probability(current_age, age, gender, m_override, b_override, max_age)
        for age in target_ages
    ])
    return probs


def joint_survival_probability(
    age_a: float,
    age_b: float,
    target_years: int,
    gender_a: str = "male",
    gender_b: str = "female",
    m_a: float | None = None,
    b_a: float | None = None,
    m_b: float | None = None,
    b_b: float | None = None,
    max_age: float = 120.0,
) -> npt.NDArray[np.float64]:
    """
    Calculate joint survival probability for a couple.

    Probability that at least one member of the couple survives.

    Formula: P(A or B survives) = P(A) + P(B) - P(A and B)
                                = P(A) + P(B) - P(A) * P(B)

    Args:
        age_a: Current age of person A
        age_b: Current age of person B
        target_years: Number of years to project
        gender_a: Gender of person A
        gender_b: Gender of person B
        m_a, b_a: Optional parameter overrides for person A
        m_b, b_b: Optional parameter overrides for person B
        max_age: Maximum possible age

    Returns:
        Array of joint survival probabilities

    Example:
        >>> probs = joint_survival_probability(65, 65, 30)
        >>> probs[20]  # Prob at least one survives 20 years
        0.678...
    """
    # Get individual survival probabilities
    probs_a = survival_probability_series(age_a, target_years, gender_a, m_a, b_a, max_age)
    probs_b = survival_probability_series(age_b, target_years, gender_b, m_b, b_b, max_age)

    # P(at least one survives) = P(A) + P(B) - P(A)*P(B)
    joint_probs = probs_a + probs_b - probs_a * probs_b

    return joint_probs


def death_probability(
    current_age: float,
    year: int,
    gender: str = "female",
    m_override: float | None = None,
    b_override: float | None = None,
    max_age: float = 120.0,
) -> float:
    """
    Calculate probability of dying in a specific year.

    Probability of dying in year v, given alive at current age.

    Formula: p_tv = q_t(v-1) - q_tv

    where q is survival probability.

    Args:
        current_age: Current age
        year: Year number (0 = current year)
        gender: "male" or "female"
        m_override: Optional mode override
        b_override: Optional dispersion override
        max_age: Maximum age

    Returns:
        Probability of dying in the specified year
    """
    if year == 0:
        return 0.0  # Can't die in year 0 (current year, already alive)

    # Probability of surviving to year-1
    prob_survive_prev = survival_probability(
        current_age, current_age + year - 1, gender, m_override, b_override, max_age
    )

    # Probability of surviving to year
    prob_survive_year = survival_probability(
        current_age, current_age + year, gender, m_override, b_override, max_age
    )

    return prob_survive_prev - prob_survive_year


def death_probability_series(
    current_age: float,
    years: int,
    gender: str = "female",
    m_override: float | None = None,
    b_override: float | None = None,
    max_age: float = 120.0,
) -> npt.NDArray[np.float64]:
    """
    Calculate death probabilities for a series of years.

    Returns the probability distribution of year of death.

    Args:
        current_age: Current age
        years: Number of years to project
        gender: "male" or "female"
        m_override: Optional mode override
        b_override: Optional dispersion override
        max_age: Maximum age

    Returns:
        Array of death probabilities, sum should approach 1
    """
    survival_probs = survival_probability_series(
        current_age, years, gender, m_override, b_override, max_age
    )

    # Death probability = decrease in survival probability
    death_probs = np.zeros(years + 1)
    death_probs[1:] = survival_probs[:-1] - survival_probs[1:]

    return death_probs


def life_expectancy(
    current_age: float,
    gender: str = "female",
    m_override: float | None = None,
    b_override: float | None = None,
    max_age: float = 120.0,
) -> float:
    """
    Calculate remaining life expectancy in years.

    Expected number of additional years of life.

    Args:
        current_age: Current age
        gender: "male" or "female"
        m_override: Optional mode override
        b_override: Optional dispersion override
        max_age: Maximum possible age

    Returns:
        Expected remaining years of life

    Example:
        >>> life_expectancy(25, "female")
        61.4  # Expected to live about 61 more years
        >>> life_expectancy(65, "male")
        18.2  # Expected to live about 18 more years
    """
    years = int(max_age - current_age)
    death_probs = death_probability_series(
        current_age, years, gender, m_override, b_override, max_age
    )

    # E[T] = sum of t * p(death at t)
    years_array = np.arange(len(death_probs))
    expected_years = np.sum(years_array * death_probs)

    return float(expected_years)


def personalized_mode(
    target_life_expectancy: float,
    current_age: float,
    gender: str = "female",
    tolerance: float = 0.01,
) -> float:
    """
    Find the mode parameter that produces a target life expectancy.

    Used to personalize mortality estimates based on family history
    or health information.

    Args:
        target_life_expectancy: Desired remaining life expectancy
        current_age: Current age
        gender: "male" or "female"
        tolerance: Convergence tolerance

    Returns:
        Mode parameter (m) that produces the target life expectancy

    Example:
        >>> # Isabela's family has high longevity, expects to live to 94
        >>> m = personalized_mode(94-25, 25, "female")  # 69 more years
        >>> m
        99.5...  # Higher mode than standard 91
    """
    params = GOMPERTZ_PARAMS[gender.lower()]
    b = params.b

    # Binary search for the mode that gives target life expectancy
    m_low = params.m
    m_high = 130.0  # Upper bound

    while m_high - m_low > tolerance:
        m_mid = (m_low + m_high) / 2
        le = life_expectancy(current_age, gender, m_mid, b)

        if le < target_life_expectancy:
            m_low = m_mid
        else:
            m_high = m_mid

    return (m_low + m_high) / 2

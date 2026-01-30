"""
Utility Functions for Lifecycle Financial Planning

Implements CRRA (Constant Relative Risk Aversion) utility functions
and the Levy-Markowitz approximation for mean-variance optimization.

Implements methodology from:
    Idzorek, T.M., & Kaplan, P.D. (2024)
    "Lifetime Financial Advice: A Personalized Optimal Multilevel Approach"
    CFA Institute Research Foundation
    With a Foreword by Roger G. Ibbotson

    Source: lifetime-financial-advice.pdf (MCP Knowledge Database)
    Primary reference: Chapter 8, Page 153

Key concepts:
    - CRRA utility: u(x) = ln(x) when θ=1, else (x^θ - 1)/θ (Equation 8.2, Page 153)
    - Risk tolerance (θ): Parameter between 0 and 1, typically. Higher values indicate more willingness to take risk
    - Levy-Markowitz: E[u(1+R)] ≈ u(1+μ) + 0.5 * u''(1+μ) * σ² (Equation 8.1, Page 153)

References:
    - Chapter 8: "Approximating Expected Utility in a Mean-Variance Model" (Pages 153-171)
    - Equation 8.1: Levy-Markowitz utility function (Page 153)
    - Equation 8.2: CRRA utility function (Page 153)
    - Equation 8.3: Second derivative of CRRA utility (Page 153)
    - Levy, H. & Markowitz, H.M. (1979). Approximating Expected Utility by a Function of Mean and Variance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PreferenceParams:
    """Financial preference parameters for an investor."""

    theta: float  # Risk tolerance (0-1, typically 0.35 for moderate)
    rho: float  # Impatience for consumption / subjective discount rate
    eta: float  # EOIS - Elasticity of intertemporal substitution
    gamma: float  # Intergenerational elasticity (bequest flexibility)
    phi: float  # Strength of bequest motive


def crra_utility(
    x: float | npt.NDArray[np.float64],
    theta: float,
) -> float | npt.NDArray[np.float64]:
    """
    Calculate CRRA (Constant Relative Risk Aversion) utility.

    The CRRA utility function has the property that relative risk aversion
    is constant regardless of wealth level.

    Formula:
        u(x) = ln(x)           if θ = 1
        u(x) = (x^θ - 1) / θ   if θ ≠ 1

    Args:
        x: Consumption level or wealth (must be positive)
        theta: Risk tolerance parameter (0 < θ ≤ 1 typically)
               - θ → 0: Extremely risk averse
               - θ = 1: Logarithmic utility (moderate risk tolerance)
               - θ > 1: High risk tolerance (less common)

    Returns:
        Utility value(s)

    Example:
        >>> crra_utility(100, 0.35)  # Low risk tolerance
        5.714...
        >>> crra_utility(100, 1.0)   # Log utility
        4.605...  # ln(100)
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        raise ValueError("Consumption/wealth x must be positive")
    if theta <= 0:
        raise ValueError("Risk tolerance theta must be positive")

    if np.isclose(theta, 1.0):
        return np.log(x)
    else:
        return (np.power(x, theta) - 1) / theta


def marginal_utility(
    x: float | npt.NDArray[np.float64],
    theta: float,
) -> float | npt.NDArray[np.float64]:
    """
    Calculate marginal utility (first derivative of CRRA utility).

    Formula: u'(x) = x^(θ-1)

    Marginal utility is always positive and decreasing (concave utility).

    Args:
        x: Consumption level or wealth
        theta: Risk tolerance parameter

    Returns:
        Marginal utility value(s)

    Example:
        >>> marginal_utility(100, 0.35)
        0.0447...  # Utility gain from extra $1
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        raise ValueError("Consumption/wealth x must be positive")

    return np.power(x, theta - 1)


def second_derivative_utility(
    x: float | npt.NDArray[np.float64],
    theta: float,
) -> float | npt.NDArray[np.float64]:
    """
    Calculate second derivative of CRRA utility.

    Formula: u''(x) = (θ-1) * x^(θ-2)

    This is always negative (concave utility = risk aversion).

    Args:
        x: Consumption level or wealth
        theta: Risk tolerance parameter

    Returns:
        Second derivative value(s), always negative
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        raise ValueError("Consumption/wealth x must be positive")

    return (theta - 1) * np.power(x, theta - 2)


def relative_risk_aversion(theta: float) -> float:
    """
    Calculate the coefficient of relative risk aversion.

    For CRRA utility, RRA = 1 - θ = constant.

    This coefficient measures how risk aversion scales with wealth.
    Higher values mean more risk averse.

    Args:
        theta: Risk tolerance parameter

    Returns:
        Relative risk aversion coefficient

    Example:
        >>> relative_risk_aversion(0.35)
        0.65  # Moderate risk aversion
        >>> relative_risk_aversion(1.0)
        0.0   # Risk neutral for gambles proportional to wealth
    """
    return 1.0 - theta


def levy_markowitz_utility(
    mu: float,
    sigma: float,
    theta: float,
) -> float:
    """
    Calculate the Levy-Markowitz approximation of expected utility.

    This approximation links expected utility theory to mean-variance
    optimization by expressing utility as a function of expected return
    and variance.

    Formula: E[u(1+R)] ≈ u(1+μ) + 0.5 * u''(1+μ) * σ²

    For CRRA: E[u(1+R)] ≈ ((1+μ)^θ - 1)/θ + 0.5 * (θ-1)*(1+μ)^(θ-2) * σ²

    The second term is always negative (since θ < 1 typically), so
    variance reduces utility.

    Args:
        mu: Expected return (e.g., 0.07 for 7%)
        sigma: Standard deviation of return
        theta: Risk tolerance parameter

    Returns:
        Approximate expected utility

    Example:
        >>> levy_markowitz_utility(0.07, 0.15, 0.35)
        0.0234...
    """
    if sigma < 0:
        raise ValueError("Standard deviation must be non-negative")

    one_plus_mu = 1 + mu

    # First term: u(1+μ)
    utility_mean = crra_utility(one_plus_mu, theta)

    # Second term: 0.5 * u''(1+μ) * σ²
    second_deriv = second_derivative_utility(one_plus_mu, theta)
    variance_penalty = 0.5 * second_deriv * sigma**2

    return float(utility_mean + variance_penalty)


def risk_adjusted_expected_return(
    mu: float,
    sigma: float,
    lambda_: float,
) -> float:
    """
    Calculate Risk-Adjusted Expected Return (RAER).

    RAER is an alternative to the Levy-Markowitz utility function
    commonly used in MVO. It's linear in expected return and
    quadratic in risk.

    Formula: RAER = μ - (λ/2) * σ²

    Args:
        mu: Expected return
        sigma: Standard deviation of return
        lambda_: Risk aversion parameter (not the same as theta!)

    Returns:
        Risk-adjusted expected return

    Note:
        λ (lambda) is the risk aversion parameter for RAER.
        It can be derived from θ using theta_to_lambda().
    """
    return mu - (lambda_ / 2) * sigma**2


def intertemporal_utility(
    consumption: npt.NDArray[np.float64],
    survival_probs: npt.NDArray[np.float64],
    rho: float,
    eta: float,
    nondiscretionary: npt.NDArray[np.float64] | None = None,
) -> float:
    """
    Calculate intertemporal utility for a consumption schedule.

    The lifetime utility function incorporates:
    - Survival probability weighting
    - Time preference (impatience) discounting
    - Preference for smooth consumption (EOIS)

    Formula: U = Σ q_tv * (1+ρ)^(-(v-t)) * u_η(c_v - c̄_v)

    where:
        q_tv = survival probability to year v
        ρ = impatience (subjective discount rate)
        η = preference for smooth consumption (EOIS)
        c_v = consumption in year v
        c̄_v = nondiscretionary consumption

    Args:
        consumption: Array of consumption amounts by year
        survival_probs: Array of survival probabilities
        rho: Subjective discount rate (impatience), e.g., 0.02 for 2%
        eta: EOIS parameter, typically 0.5
        nondiscretionary: Optional array of nondiscretionary consumption

    Returns:
        Total intertemporal utility

    Example:
        >>> consumption = np.array([50000, 51000, 52000])  # Growing consumption
        >>> survival = np.array([1.0, 0.99, 0.98])
        >>> intertemporal_utility(consumption, survival, 0.02, 0.5)
        145234...
    """
    T = len(consumption)

    if len(survival_probs) != T:
        raise ValueError("consumption and survival_probs must have same length")

    if nondiscretionary is None:
        nondiscretionary = np.zeros(T)
    elif len(nondiscretionary) != T:
        raise ValueError("nondiscretionary must have same length as consumption")

    # Discretionary consumption
    discretionary = consumption - nondiscretionary

    if np.any(discretionary <= 0):
        raise ValueError("Discretionary consumption must be positive")

    # Calculate utility for each period
    total_utility = 0.0
    for v in range(T):
        # Survival probability weight
        q = survival_probs[v]

        # Time discount factor
        discount = (1 + rho) ** (-v)

        # Period utility (using eta as the preference parameter)
        period_utility = crra_utility(discretionary[v], eta)

        total_utility += q * discount * period_utility

    return float(total_utility)


def bequest_utility(
    bequest: float,
    gamma: float,
    phi: float,
) -> float:
    """
    Calculate utility from leaving a bequest.

    The bequest utility function determines how much satisfaction
    an investor gets from leaving wealth to heirs.

    Formula: v(B) = φ * u_γ(B) = φ * (B^γ - 1) / γ

    where:
        B = bequest amount
        γ (gamma) = intergenerational elasticity (bequest flexibility)
        φ (phi) = strength of bequest motive

    Args:
        bequest: Amount left as bequest
        gamma: Intergenerational elasticity (0 to 1)
        phi: Strength of bequest motive

    Returns:
        Utility from bequest

    Example:
        >>> bequest_utility(500000, 0.25, 1.5)
        39.87...
    """
    if bequest <= 0:
        return 0.0

    return phi * crra_utility(bequest, gamma)


def combined_utility(
    consumption: npt.NDArray[np.float64],
    bequest: float,
    survival_probs: npt.NDArray[np.float64],
    death_probs: npt.NDArray[np.float64],
    rho: float,
    eta: float,
    gamma: float,
    phi: float,
    nondiscretionary: npt.NDArray[np.float64] | None = None,
) -> float:
    """
    Calculate combined utility from consumption and bequest.

    Combines intertemporal consumption utility with bequest utility,
    weighting bequest by probability of death in each year.

    Formula: U_total = U_consumption + Σ p_tv * (1+ρ)^(-(v-t)) * φ * u_γ(B_v)

    where p_tv is the probability of dying in year v.

    Args:
        consumption: Array of consumption by year
        bequest: Bequest amount (assumed constant for simplicity)
        survival_probs: Survival probabilities by year
        death_probs: Death probabilities by year
        rho: Subjective discount rate
        eta: EOIS parameter
        gamma: Intergenerational elasticity
        phi: Bequest motive strength
        nondiscretionary: Optional nondiscretionary consumption

    Returns:
        Total utility from consumption and bequest
    """
    # Consumption utility
    u_consumption = intertemporal_utility(
        consumption, survival_probs, rho, eta, nondiscretionary
    )

    # Bequest utility (weighted by death probability)
    u_bequest = 0.0
    for v in range(len(death_probs)):
        if death_probs[v] > 0:
            discount = (1 + rho) ** (-v)
            u_bequest += death_probs[v] * discount * bequest_utility(bequest, gamma, phi)

    return u_consumption + u_bequest


def certainty_equivalent(
    expected_utility: float,
    theta: float,
) -> float:
    """
    Calculate certainty equivalent from expected utility.

    The certainty equivalent is the guaranteed amount that provides
    the same utility as the risky prospect.

    Formula: CE = u^(-1)(EU) = (θ * EU + 1)^(1/θ)  for θ ≠ 1
             CE = exp(EU)                          for θ = 1

    Args:
        expected_utility: Expected utility value
        theta: Risk tolerance parameter

    Returns:
        Certainty equivalent consumption/wealth

    Example:
        >>> eu = levy_markowitz_utility(0.07, 0.15, 0.35)
        >>> certainty_equivalent(eu, 0.35)
        1.024...  # CE is 2.4% return
    """
    if np.isclose(theta, 1.0):
        return np.exp(expected_utility)
    else:
        return np.power(theta * expected_utility + 1, 1 / theta)

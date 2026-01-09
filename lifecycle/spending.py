"""
Optimal Spending and Consumption Rules

Implements optimal spending rules for lifecycle financial planning:
    - Deterministic spending rules (no market risk)
    - Stochastic spending rules (with market risk)
    - With and without annuities

Based on Chapters 5 and 6 of "Lifetime Financial Advice".

Key concepts:
    - Discretionary consumption = Net worth / Divisor
    - Divisor depends on preferences and survival probabilities
    - Annuities affect mortality weighting of calculations

References:
    - Waring, M.B. & Siegel, L.B. (2015). The Only Spending Rule You'll Ever Need.
    - Kaplan, P. (2015). More realistic spending rules.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def consumption_growth_rate(
    risk_free_rate: float,
    rho: float,
    eta: float,
) -> float:
    """
    Calculate optimal consumption growth rate.

    In the deterministic case, optimal consumption grows at a rate
    determined by the relationship between impatience and the market.

    Formula: g = (r - ρ) / (1 - η)

    where:
        r = risk-free rate
        ρ = impatience (subjective discount rate)
        η = preference for smooth consumption (EOIS)

    Args:
        risk_free_rate: Risk-free rate of return
        rho: Impatience / subjective discount rate
        eta: EOIS parameter (preference for smooth consumption)

    Returns:
        Optimal annual consumption growth rate

    Example:
        >>> consumption_growth_rate(0.025, 0.02, 0.5)
        0.01  # 1% growth when r > ρ (patient investor)
        >>> consumption_growth_rate(0.025, 0.05, 0.5)
        -0.05  # Negative growth when ρ > r (impatient investor)
    """
    if eta >= 1.0:
        raise ValueError("eta must be less than 1")
    return (risk_free_rate - rho) / (1 - eta)


def optimal_consumption_divisor(
    years: int,
    risk_free_rate: float,
    rho: float,
    eta: float,
    survival_probs: npt.NDArray[np.float64] | None = None,
) -> float:
    """
    Calculate the divisor for optimal discretionary consumption.

    Discretionary consumption = Net Worth / Divisor

    The divisor captures:
        - Time value of money
        - Survival probabilities
        - Preference parameters

    Formula (base case without annuities):
        Δ = Σ q_tv * (1+g)^(v-t) / (1+r)^(v-t)

    where g is the consumption growth rate.

    Args:
        years: Planning horizon in years
        risk_free_rate: Risk-free rate
        rho: Subjective discount rate (impatience)
        eta: EOIS parameter
        survival_probs: Survival probabilities by year (optional)

    Returns:
        Consumption divisor

    Example:
        >>> divisor = optimal_consumption_divisor(40, 0.025, 0.02, 0.5)
        >>> discretionary = 1000000 / divisor
        >>> discretionary
        35714...  # Annual discretionary consumption
    """
    g = consumption_growth_rate(risk_free_rate, rho, eta)

    if survival_probs is None:
        survival_probs = np.ones(years)

    divisor = 0.0
    for v in range(years):
        growth_factor = (1 + g) ** v
        discount_factor = (1 + risk_free_rate) ** (-v)
        divisor += survival_probs[v] * growth_factor * discount_factor

    return divisor


def optimal_consumption_divisor_with_annuities(
    years: int,
    risk_free_rate: float,
    rho: float,
    eta: float,
    survival_probs: npt.NDArray[np.float64],
    retirement_year: int,
) -> float:
    """
    Calculate consumption divisor when annuities are available at retirement.

    Before retirement, survival probabilities don't affect divisor (no annuities).
    After retirement, mortality-weighted divisor applies.

    Args:
        years: Planning horizon in years
        risk_free_rate: Risk-free rate
        rho: Subjective discount rate
        eta: EOIS parameter
        survival_probs: Survival probabilities by year
        retirement_year: Year of retirement (0-indexed from current year)

    Returns:
        Consumption divisor with annuity adjustment
    """
    g = consumption_growth_rate(risk_free_rate, rho, eta)

    # Create adjusted survival probabilities
    # Before retirement: q = 1 (no mortality weighting)
    # After retirement: q = actual survival probability
    adjusted_probs = np.ones(years)
    if retirement_year < years:
        # Ensure survival_probs has the right length (should be years, not years+1)
        # If survival_probs has length years+1, use only the first 'years' elements
        if len(survival_probs) == years + 1:
            survival_probs_adjusted = survival_probs[:-1]  # Drop the last element
        else:
            survival_probs_adjusted = survival_probs
        
        # Renormalize post-retirement probabilities
        adjusted_probs[retirement_year:] = (
            survival_probs_adjusted[retirement_year:]
            / survival_probs_adjusted[retirement_year]
        )

    divisor = 0.0
    for v in range(years):
        growth_factor = (1 + g) ** v
        discount_factor = (1 + risk_free_rate) ** (-v)
        divisor += adjusted_probs[v] * growth_factor * discount_factor

    return divisor


def discretionary_consumption(
    net_worth: float,
    divisor: float,
) -> float:
    """
    Calculate optimal discretionary consumption.

    Formula: c = W / Δ

    where W is net worth and Δ is the consumption divisor.

    Args:
        net_worth: Current net worth (financial + human capital - liabilities)
        divisor: Consumption divisor from optimal_consumption_divisor()

    Returns:
        Optimal annual discretionary consumption

    Example:
        >>> discretionary_consumption(1475625, 28.0)
        52701.0
    """
    return net_worth / divisor


def rescheduling_factor(
    year: int,
    base_year: int,
    survival_probs: npt.NDArray[np.float64],
    eta: float,
) -> float:
    """
    Calculate rescheduling factor for mortality-adjusted spending.

    The rescheduling factor indicates how much spending to move
    from later years to nearer years because of mortality risk.

    Formula: RF_v = (q_tv)^(1/(1-η))

    Lower η means less rescheduling (more smoothing preferred).

    Args:
        year: Target year (v)
        base_year: Current year (t)
        survival_probs: Survival probabilities
        eta: EOIS parameter

    Returns:
        Rescheduling factor (0 to 1)

    Example:
        >>> # With low eta (0.2), investor wants smooth consumption
        >>> rescheduling_factor(20, 0, probs, 0.2)
        0.92  # Small adjustment
        >>> # With high eta (0.8), more willing to adjust
        >>> rescheduling_factor(20, 0, probs, 0.8)
        0.65  # Larger adjustment
    """
    if year < 0 or year >= len(survival_probs):
        return 0.0

    q = survival_probs[year]
    if q <= 0:
        return 0.0

    exponent = 1 / (1 - eta)
    return q ** exponent


def spending_rule_no_annuities(
    net_worth: float,
    years_remaining: int,
    risk_free_rate: float,
    rho: float,
    eta: float,
    survival_probs: npt.NDArray[np.float64],
    nondiscretionary: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate spending using the rule without annuities.

    Returns both discretionary and total consumption.

    The spending rule says:
        1. Calculate your net worth
        2. Calculate the divisor
        3. Divide to get discretionary consumption
        4. Add nondiscretionary to get total spending

    Args:
        net_worth: Current net worth
        years_remaining: Years in planning horizon
        risk_free_rate: Risk-free rate
        rho: Subjective discount rate
        eta: EOIS parameter
        survival_probs: Survival probabilities
        nondiscretionary: Annual nondiscretionary consumption

    Returns:
        Tuple of (discretionary_consumption, total_consumption)

    Example:
        >>> disc, total = spending_rule_no_annuities(
        ...     1475625, 40, 0.025, 0.02, 0.5, probs, 40000
        ... )
        >>> disc
        52701.0
        >>> total
        92701.0  # Includes nondiscretionary
    """
    divisor = optimal_consumption_divisor(
        years_remaining, risk_free_rate, rho, eta, survival_probs
    )
    disc = discretionary_consumption(net_worth, divisor)
    return disc, disc + nondiscretionary


def spending_rule_with_annuities(
    net_worth: float,
    years_remaining: int,
    risk_free_rate: float,
    rho: float,
    eta: float,
    survival_probs: npt.NDArray[np.float64],
    retirement_year: int,
    nondiscretionary: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate spending using the rule with annuities at retirement.

    Similar to no-annuities case but uses mortality-adjusted divisor.

    Args:
        net_worth: Current net worth
        years_remaining: Years in planning horizon
        risk_free_rate: Risk-free rate
        rho: Subjective discount rate
        eta: EOIS parameter
        survival_probs: Survival probabilities
        retirement_year: Year of retirement
        nondiscretionary: Annual nondiscretionary consumption

    Returns:
        Tuple of (discretionary_consumption, total_consumption)
    """
    divisor = optimal_consumption_divisor_with_annuities(
        years_remaining, risk_free_rate, rho, eta,
        survival_probs, retirement_year
    )
    disc = discretionary_consumption(net_worth, divisor)
    return disc, disc + nondiscretionary


def waring_siegel_spending(
    financial_wealth: float,
    real_return: float,
    years_remaining: int,
) -> float:
    """
    Calculate spending using the Waring-Siegel rule.

    A simplified spending rule that divides wealth by remaining years,
    adjusted for expected returns.

    Formula: Spending = W / ΣPVIFA

    where PVIFA is the present value interest factor of an annuity.

    This is equivalent to amortizing wealth over remaining lifetime.

    Args:
        financial_wealth: Current financial assets
        real_return: Expected real return
        years_remaining: Expected remaining years

    Returns:
        Annual spending amount

    Example:
        >>> waring_siegel_spending(1000000, 0.03, 30)
        50193.0  # About 5% of wealth
    """
    if years_remaining <= 0:
        return financial_wealth

    # PVIFA = (1 - (1+r)^-n) / r
    if np.isclose(real_return, 0):
        pvifa = years_remaining
    else:
        pvifa = (1 - (1 + real_return) ** (-years_remaining)) / real_return

    return financial_wealth / pvifa


def kaplan_blanchett_spending(
    net_worth: float,
    years_remaining: int,
    certainty_equivalent_return: float,
    rho: float,
    eta: float,
    survival_probs: npt.NDArray[np.float64],
) -> float:
    """
    Calculate spending using the Kaplan-Blanchett model.

    Extends Waring-Siegel by incorporating:
        - Survival probabilities
        - Investor preferences
        - Certainty equivalent return

    This is the full stochastic spending rule from Chapter 6.

    Args:
        net_worth: Current net worth
        years_remaining: Years in planning horizon
        certainty_equivalent_return: Certainty equivalent return (accounts for risk)
        rho: Subjective discount rate
        eta: EOIS parameter
        survival_probs: Survival probabilities

    Returns:
        Optimal annual discretionary spending
    """
    g = consumption_growth_rate(certainty_equivalent_return, rho, eta)

    divisor = 0.0
    for v in range(years_remaining):
        rf = rescheduling_factor(v, 0, survival_probs, eta)
        growth_factor = (1 + g) ** v
        discount_factor = (1 + certainty_equivalent_return) ** (-v)
        divisor += rf * growth_factor * discount_factor

    return net_worth / divisor


def consumption_schedule(
    initial_discretionary: float,
    years: int,
    growth_rate: float,
    nondiscretionary: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate complete consumption schedule from initial consumption.

    Args:
        initial_discretionary: First year discretionary consumption
        years: Number of years
        growth_rate: Annual consumption growth rate
        nondiscretionary: Array of nondiscretionary consumption by year

    Returns:
        Tuple of (discretionary_array, total_array)
    """
    discretionary = initial_discretionary * (1 + growth_rate) ** np.arange(years)

    if nondiscretionary is None:
        nondiscretionary = np.zeros(years)

    total = discretionary + nondiscretionary

    return discretionary, total


def sustainable_spending_rate(
    years_remaining: int,
    real_return: float,
    failure_probability: float = 0.05,
    return_volatility: float = 0.10,
) -> float:
    """
    Calculate sustainable spending rate given failure probability.

    Uses simplified Monte Carlo approximation to find the spending
    rate that has a given probability of portfolio exhaustion.

    Args:
        years_remaining: Expected remaining years
        real_return: Expected real return
        failure_probability: Acceptable probability of running out (default 5%)
        return_volatility: Standard deviation of returns

    Returns:
        Sustainable annual withdrawal rate (as decimal)

    Example:
        >>> sustainable_spending_rate(30, 0.05, 0.05, 0.15)
        0.0395  # ~4% rule
    """
    # Approximate using safety-first approach
    # Lower bound return = mean - z * std
    z = 1.645 if failure_probability == 0.05 else 1.28  # 5% or 10%

    safe_return = real_return - z * return_volatility

    if safe_return <= 0:
        safe_return = 0.01  # Floor

    # Use annuity factor with safe return
    if np.isclose(safe_return, 0):
        rate = 1 / years_remaining
    else:
        pvifa = (1 - (1 + safe_return) ** (-years_remaining)) / safe_return
        rate = 1 / pvifa

    return rate


def simulate_spending_paths(
    initial_wealth: float,
    initial_spending: float,
    spending_growth: float,
    expected_return: float,
    return_volatility: float,
    years: int,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simulate wealth and spending paths under uncertainty.

    Args:
        initial_wealth: Starting wealth
        initial_spending: First year spending
        spending_growth: Annual spending growth rate
        expected_return: Expected investment return
        return_volatility: Return standard deviation
        years: Number of years to simulate
        n_simulations: Number of simulation paths
        seed: Random seed

    Returns:
        Tuple of (wealth_paths, spending_paths) arrays
    """
    if seed is not None:
        np.random.seed(seed)

    wealth_paths = np.zeros((n_simulations, years + 1))
    spending_paths = np.zeros((n_simulations, years))

    wealth_paths[:, 0] = initial_wealth

    for t in range(years):
        # Current spending
        spending = initial_spending * (1 + spending_growth) ** t
        spending_paths[:, t] = spending

        # Investment returns (lognormal)
        log_returns = np.random.normal(
            np.log(1 + expected_return) - return_volatility**2 / 2,
            return_volatility,
            n_simulations
        )
        returns = np.exp(log_returns) - 1

        # Wealth evolution: W_{t+1} = (W_t - S_t) * (1 + R_t)
        wealth_after_spending = np.maximum(wealth_paths[:, t] - spending, 0)
        wealth_paths[:, t + 1] = wealth_after_spending * (1 + returns)

    return wealth_paths, spending_paths


def failure_probability(
    wealth_paths: npt.NDArray[np.float64],
) -> float:
    """
    Calculate probability of portfolio failure from simulations.

    Portfolio failure is defined as wealth reaching zero.

    Args:
        wealth_paths: Array of simulated wealth paths

    Returns:
        Probability of failure (0 to 1)
    """
    failures = np.any(wealth_paths <= 0, axis=1)
    return np.mean(failures)

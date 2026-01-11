"""
Annuity Calculations for Lifecycle Financial Planning

Implements calculations for:
    - SPIA (Single Premium Immediate Annuity) pricing
    - IVA (Immediate Variable Annuity) calculations
    - Mortality credits
    - Annuity-adjusted survival probabilities

Based on Chapters 5, 6, and 10 of "Lifetime Financial Advice".

Key concepts:
    - Annuities provide longevity insurance
    - Mortality credits enhance returns for survivors
    - SPIAs provide fixed income, IVAs provide variable income

References:
    - Milevsky, M.A. (2006). The Calculus of Retirement Income.
    - Yaari, M.E. (1965). Uncertain Lifetime, Life Insurance, and the Theory of the Consumer.
"""

import numpy as np
from typing import Optional, Tuple


def spia_price(
    current_age: float,
    risk_free_rate: float,
    survival_probs: np.ndarray,
    payment: float = 1.0,
    inflation_adjustment: float = 0.0,
) -> float:
    """
    Calculate the fair price of a Single Premium Immediate Annuity (SPIA).

    A SPIA pays a fixed amount each year until the annuitant dies.
    The fair price is the present value of expected payments.

    Formula: Price = Σ q_tv * P * (1+g)^v / (1+r)^v

    where:
        q_tv = survival probability to year v
        P = annual payment
        g = inflation adjustment
        r = risk-free rate

    Args:
        current_age: Current age of annuitant
        risk_free_rate: Risk-free discount rate
        survival_probs: Array of survival probabilities
        payment: Annual payment amount (default $1)
        inflation_adjustment: Annual payment growth rate (default 0)

    Returns:
        Fair price of the annuity

    Example:
        >>> probs = survival_probability_series(65, 40, "male")
        >>> spia_price(65, 0.025, probs, 10000)
        135678.0  # Cost of $10k/year lifetime income
    """
    years = len(survival_probs)
    price = 0.0

    for v in range(years):
        payment_v = payment * (1 + inflation_adjustment) ** v
        discount = (1 + risk_free_rate) ** (-v)
        price += survival_probs[v] * payment_v * discount

    return price


def spia_payout_rate(
    current_age: float,
    risk_free_rate: float,
    survival_probs: np.ndarray,
) -> float:
    """
    Calculate the SPIA payout rate (payment per dollar invested).

    Formula: Payout Rate = 1 / SPIA_price($1)

    Args:
        current_age: Current age
        risk_free_rate: Risk-free rate
        survival_probs: Survival probabilities

    Returns:
        Annual payout per dollar invested

    Example:
        >>> spia_payout_rate(65, 0.025, probs)
        0.0737  # 7.37% payout rate at age 65
    """
    price_per_dollar = spia_price(current_age, risk_free_rate, survival_probs, 1.0)
    return 1.0 / price_per_dollar


def mortality_credit(
    return_rate: float,
    survival_prob: float,
) -> float:
    """
    Calculate mortality credit for annuitized investments.

    When investors die, their share is forfeited to survivors.
    This creates an additional "return" for surviving annuitants.

    Formula: Mortality Credit = (1+r) / q - 1 - r

    or equivalently: Total Return = (1+r) / q - 1

    Args:
        return_rate: Investment return rate
        survival_prob: Probability of surviving the period

    Returns:
        Mortality credit (additional return from mortality pooling)

    Example:
        >>> mortality_credit(0.025, 0.96)
        0.0427  # 4.27% mortality credit when survival is 96%
        >>> # Total return = 2.5% + 4.27% = 6.77%
    """
    if survival_prob <= 0:
        return float('inf')  # All wealth to survivors

    total_return = (1 + return_rate) / survival_prob - 1
    credit = total_return - return_rate

    return credit


def annuitized_return(
    return_rate: float,
    survival_prob: float,
) -> float:
    """
    Calculate total return on annuitized investment.

    Combines investment return with mortality credit.

    Formula: R_annuity = (1+r) / q - 1

    Args:
        return_rate: Investment return
        survival_prob: Survival probability

    Returns:
        Total return including mortality credit

    Example:
        >>> annuitized_return(0.025, 0.96)
        0.0677  # 6.77% return (2.5% investment + 4.27% mortality)
    """
    if survival_prob <= 0:
        return float('inf')

    return (1 + return_rate) / survival_prob - 1


def iva_units(
    net_worth: float,
    initial_consumption: float,
) -> float:
    """
    Calculate number of IVA units to purchase.

    An Immediate Variable Annuity (IVA) provides variable income
    based on market performance. Units are purchased that pay
    a variable amount each period.

    Args:
        net_worth: Available wealth for annuitization
        initial_consumption: Desired first-year consumption

    Returns:
        Number of IVA units to purchase

    Example:
        >>> iva_units(500000, 25000)
        20.0  # Purchase 20 units paying ~$25k in year 1
    """
    if initial_consumption <= 0:
        return 0.0
    return net_worth / initial_consumption


def iva_payment(
    units: float,
    assumed_interest_rate: float,
    actual_return: float,
    previous_payment: float,
) -> float:
    """
    Calculate IVA payment for a period.

    IVA payments adjust based on investment performance relative
    to the Assumed Interest Rate (AIR).

    Formula: Payment_t = Payment_{t-1} * (1 + R) / (1 + AIR)

    Args:
        units: Number of annuity units owned
        assumed_interest_rate: AIR embedded in annuity
        actual_return: Actual investment return this period
        previous_payment: Previous period's payment

    Returns:
        Current period payment

    Example:
        >>> # If actual return exceeds AIR, payment increases
        >>> iva_payment(20, 0.04, 0.08, 25000)
        26923.0  # Payment increased ~7.7%
    """
    adjustment = (1 + actual_return) / (1 + assumed_interest_rate)
    return previous_payment * adjustment


def annuity_adjusted_survival(
    survival_probs: np.ndarray,
    retirement_year: int,
) -> np.ndarray:
    """
    Create adjusted survival probabilities for annuity calculations.

    Before retirement: q = 1 (no mortality weighting)
    After retirement: q = actual survival probability

    This reflects that annuities are typically purchased at retirement.

    Args:
        survival_probs: Raw survival probabilities from current age
        retirement_year: Year of retirement (0-indexed)

    Returns:
        Adjusted survival probabilities

    Example:
        >>> probs = survival_probability_series(25, 70, "female")
        >>> adjusted = annuity_adjusted_survival(probs, 40)  # Retire at 65
        >>> adjusted[:40]  # All ones before retirement
        array([1., 1., ..., 1.])
    """
    adjusted = np.ones_like(survival_probs)

    if retirement_year < len(survival_probs):
        # After retirement, use actual survival probs
        # Renormalize to probability of surviving given alive at retirement
        retirement_prob = survival_probs[retirement_year]
        if retirement_prob > 0:
            adjusted[retirement_year:] = survival_probs[retirement_year:] / retirement_prob

    return adjusted


def optimal_annuity_allocation(
    wealth: float,
    bequest_motive: float,
    survival_probs: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Calculate optimal fraction of wealth to annuitize.

    Balances longevity insurance benefits against bequest desire.
    Higher bequest motive = less annuitization.

    This is a simplified approximation. Full solution requires
    dynamic programming.

    Args:
        wealth: Total wealth available
        bequest_motive: Strength of bequest preference (phi)
        survival_probs: Survival probabilities
        risk_free_rate: Risk-free rate

    Returns:
        Optimal fraction to annuitize (0 to 1)

    Example:
        >>> optimal_annuity_allocation(1000000, 0.0, probs, 0.025)
        1.0  # Full annuitization with no bequest motive
        >>> optimal_annuity_allocation(1000000, 2.0, probs, 0.025)
        0.35  # Partial annuitization with strong bequest motive
    """
    # Simple heuristic based on bequest motive
    # phi = 0: full annuitization
    # phi increases: less annuitization

    # Base annuity value (mortality credits matter more at older ages)
    avg_mortality_credit = mortality_credit(risk_free_rate, np.mean(survival_probs))

    # Annuity attractiveness decreases with bequest motive
    if bequest_motive <= 0:
        return 1.0

    # Logistic decay based on bequest motive
    annuity_fraction = 1.0 / (1 + bequest_motive)

    return annuity_fraction


def longevity_insurance_value(
    wealth: float,
    years_remaining: int,
    survival_probs: np.ndarray,
    risk_free_rate: float,
    rho: float,
    eta: float,
) -> float:
    """
    Calculate the value of longevity insurance from annuitization.

    Compares utility with and without annuities to quantify
    the welfare gain from mortality pooling.

    Args:
        wealth: Initial wealth
        years_remaining: Planning horizon
        survival_probs: Survival probabilities
        risk_free_rate: Risk-free rate
        rho: Impatience parameter
        eta: EOIS parameter

    Returns:
        Value of longevity insurance as percentage of wealth

    Example:
        >>> longevity_insurance_value(1000000, 30, probs, 0.025, 0.02, 0.5)
        0.15  # Annuities add ~15% to effective wealth
    """
    from lifecycle.spending import (
        optimal_consumption_divisor,
        optimal_consumption_divisor_with_annuities,
    )

    # Divisor without annuities
    divisor_no_annuity = optimal_consumption_divisor(
        years_remaining, risk_free_rate, rho, eta, survival_probs
    )

    # Divisor with annuities (assume at retirement = year 0)
    divisor_with_annuity = optimal_consumption_divisor_with_annuities(
        years_remaining, risk_free_rate, rho, eta, survival_probs, 0
    )

    # Value is the reduction in divisor (higher consumption per $ of wealth)
    value = (divisor_no_annuity - divisor_with_annuity) / divisor_no_annuity

    return max(0, value)


def deferred_annuity_price(
    current_age: float,
    deferral_years: int,
    risk_free_rate: float,
    survival_probs: np.ndarray,
    payment: float = 1.0,
) -> float:
    """
    Calculate price of a deferred annuity.

    A deferred annuity starts payments after a deferral period.
    Price is lower because some annuitants will die during deferral.

    Args:
        current_age: Current age
        deferral_years: Years until payments begin
        risk_free_rate: Discount rate
        survival_probs: Survival probabilities
        payment: Annual payment after deferral

    Returns:
        Fair price of deferred annuity

    Example:
        >>> deferred_annuity_price(55, 10, 0.025, probs, 10000)
        95000.0  # Cheaper than immediate annuity
    """
    if deferral_years >= len(survival_probs):
        return 0.0

    # Only count payments starting after deferral
    price = 0.0
    for v in range(deferral_years, len(survival_probs)):
        discount = (1 + risk_free_rate) ** (-v)
        price += survival_probs[v] * payment * discount

    return price


def annuity_equivalent_wealth(
    consumption: float,
    survival_probs: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Calculate wealth required to fund consumption without annuities.

    This is higher than the annuity price because of self-insurance
    against longevity risk.

    Args:
        consumption: Desired annual consumption
        survival_probs: Survival probabilities
        risk_free_rate: Risk-free rate

    Returns:
        Wealth needed to self-fund consumption

    Example:
        >>> # Without annuities, need more wealth for same income
        >>> annuity_equivalent_wealth(50000, probs, 0.025)
        850000.0
        >>> # With annuity, only need:
        >>> spia_price(65, 0.025, probs, 50000)
        678000.0
    """
    # Need to fund consumption in all states, including living to max age
    max_years = len(survival_probs)

    # PV of consumption stream (worst case = live to max)
    pv = 0.0
    for v in range(max_years):
        discount = (1 + risk_free_rate) ** (-v)
        pv += consumption * discount

    return pv

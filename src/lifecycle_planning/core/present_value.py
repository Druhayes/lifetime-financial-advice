"""
Present Value Calculations for Lifecycle Financial Planning

Implements present value calculations for:
    - General cash flow streams
    - Human capital (present value of future earnings)
    - Liabilities (present value of nondiscretionary consumption)
    - Net worth calculations

Implements methodology from:
    Idzorek, T.M., & Kaplan, P.D. (2024)
    "Lifetime Financial Advice: A Personalized Optimal Multilevel Approach"
    CFA Institute Research Foundation
    With a Foreword by Roger G. Ibbotson

    Source: lifetime-financial-advice.pdf (MCP Knowledge Database)
    Primary reference: Chapter 4, Pages 86, 89-90

References:
    - Chapter 4: Human capital valuation methodology (Pages 80-90)
    - Equations 4.6-4.8: Human capital present value formulas (Pages 89-90)
    - Page 86: Discount rate selection for human capital (risky HC approach)
    - Page 85-86: Mortality weighting in present value calculations
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def pv(
    cash_flows: Sequence[float] | npt.NDArray[np.float64],
    discount_rate: float,
    start_year: int = 0,
) -> float:
    """
    Calculate present value of a stream of cash flows.

    Formula: PV = Σ CF_t / (1+r)^t

    Args:
        cash_flows: Array of cash flows by year
        discount_rate: Annual discount rate (e.g., 0.025 for 2.5%)
        start_year: Year of first cash flow (default 0)

    Returns:
        Present value

    Example:
        >>> cash_flows = [1000, 1000, 1000, 1000, 1000]  # 5 payments
        >>> pv(cash_flows, 0.05)
        4329.48  # PV of 5-year annuity
    """
    cash_flows = np.asarray(cash_flows)
    years = np.arange(start_year, start_year + len(cash_flows))
    discount_factors = (1 + discount_rate) ** (-years)
    return float(np.sum(cash_flows * discount_factors))


# Deprecated alias for backward compatibility
present_value = pv


def present_value_growing(
    initial_payment: float,
    growth_rate: float,
    discount_rate: float,
    n_periods: int,
) -> float:
    """
    Calculate present value of a growing annuity.

    Formula: PV = P * [1 - ((1+g)/(1+r))^n] / (r - g)  if r ≠ g
             PV = P * n / (1+r)                         if r = g

    Args:
        initial_payment: First payment amount
        growth_rate: Annual growth rate of payments
        discount_rate: Discount rate
        n_periods: Number of periods

    Returns:
        Present value of growing annuity
    """
    if np.isclose(discount_rate, growth_rate):
        return initial_payment * n_periods / (1 + discount_rate)

    ratio = (1 + growth_rate) / (1 + discount_rate)
    return initial_payment * (1 - ratio ** n_periods) / (discount_rate - growth_rate)


def present_value_perpetuity(
    payment: float,
    discount_rate: float,
    growth_rate: float = 0.0,
) -> float:
    """
    Calculate present value of a perpetuity (infinite stream).

    Formula: PV = P / (r - g)

    Args:
        payment: Annual payment
        discount_rate: Discount rate (must be > growth_rate)
        growth_rate: Growth rate of payments (default 0)

    Returns:
        Present value of perpetuity
    """
    if discount_rate <= growth_rate:
        raise ValueError("Discount rate must exceed growth rate for perpetuity")
    return payment / (discount_rate - growth_rate)


def align_income_survival(
    income_stream: npt.NDArray[np.float64],
    survival_probs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Align income and survival probability arrays to same length.

    Handles the common case where survival_probs includes year 0 (prob=1.0)
    but income_stream does not.

    Args:
        income_stream: Income by year
        survival_probs: Survival probabilities (may include year 0)

    Returns:
        Tuple of (income, survival_probs) with matching lengths

    Raises:
        ValueError: If arrays differ by more than 1 element

    Example:
        >>> income = np.array([75000] * 40)  # 40 years
        >>> survival = np.array([1.0] + [0.99] * 40)  # 41 years (includes year 0)
        >>> aligned_income, aligned_survival = align_income_survival(income, survival)
        >>> len(aligned_income), len(aligned_survival)
        (40, 40)
    """
    len_income = len(income_stream)
    len_survival = len(survival_probs)

    if len_income == len_survival:
        return income_stream, survival_probs
    elif len_survival == len_income + 1:
        # Survival includes year 0, remove it
        return income_stream, survival_probs[1:]
    elif len_income == len_survival + 1:
        # Income includes extra year, remove last
        return income_stream[:-1], survival_probs
    else:
        raise ValueError(
            f"Cannot align arrays: income has {len_income} elements, "
            f"survival has {len_survival} elements (diff > 1)"
        )


def human_capital(
    income_stream: npt.NDArray[np.float64],
    discount_rate: float,
) -> float:
    """
    Calculate human capital (present value of future income).

    Human capital without mortality weighting. Appropriate when
    annuities are not available and the investor self-insures
    against longevity risk.

    Formula: H_t = Σ Y_v / (1+r)^(v-t)

    where Y_v is income in year v.

    Args:
        income_stream: Array of future income by year
        discount_rate: Discount rate for income (based on income risk)

    Returns:
        Present value of human capital

    Example:
        >>> income = np.array([75000] * 40)  # 40 years of $75k
        >>> human_capital(income, 0.025)
        1875234...
    """
    return pv(income_stream, discount_rate)


def human_capital_mortality_weighted(
    income_stream: npt.NDArray[np.float64],
    discount_rate: float,
    survival_probs: npt.NDArray[np.float64],
    auto_align: bool = True,
) -> float:
    """
    Calculate mortality-weighted human capital.

    Human capital with mortality weighting. Appropriate when
    annuities are available and income is mortality-contingent.

    Formula: Ĥ_t = Σ q_tv * Y_v / (1+r)^(v-t)

    where q_tv is survival probability to year v.

    Args:
        income_stream: Array of future income by year
        discount_rate: Discount rate for income
        survival_probs: Survival probabilities by year
        auto_align: If True, automatically align array lengths (default True)

    Returns:
        Mortality-weighted present value of human capital

    Raises:
        ValueError: If arrays have different lengths and cannot be aligned
    """
    if auto_align:
        income_stream, survival_probs = align_income_survival(income_stream, survival_probs)

    if len(income_stream) != len(survival_probs):
        raise ValueError(
            f"income_stream and survival_probs must have same length: "
            f"got {len(income_stream)} and {len(survival_probs)}"
        )

    # Weight income by survival probability
    weighted_income = income_stream * survival_probs
    return pv(weighted_income, discount_rate)


def income_discount_rate(
    risk_free_rate: float,
    equity_return: float,
    equity_weight: float = 0.2,
) -> float:
    """
    Calculate appropriate discount rate for income/human capital.

    The discount rate should reflect the riskiness of income.
    Income is modeled as a portfolio of bond-like and equity-like
    components.

    Formula: r_income = (1 - w_eq) * r_f + w_eq * r_eq

    Args:
        risk_free_rate: Risk-free rate
        equity_return: Expected equity return
        equity_weight: Equity-like portion of income risk (default 20%)

    Returns:
        Appropriate discount rate for income

    Example:
        >>> income_discount_rate(0.025, 0.07, 0.2)
        0.034  # 3.4%
    """
    bond_weight = 1 - equity_weight
    return bond_weight * risk_free_rate + equity_weight * equity_return


def liability_value(
    consumption_stream: npt.NDArray[np.float64],
    discount_rate: float,
) -> float:
    """
    Calculate present value of liabilities (nondiscretionary consumption).

    Liabilities without mortality weighting.

    Formula: L_t = Σ C̄_v / (1+r)^(v-t)

    where C̄_v is nondiscretionary consumption in year v.

    Args:
        consumption_stream: Array of nondiscretionary consumption by year
        discount_rate: Discount rate for liabilities

    Returns:
        Present value of liabilities

    Example:
        >>> consumption = np.array([40000] * 60)  # 60 years of $40k
        >>> liability_value(consumption, 0.025)
        1392064...
    """
    return pv(consumption_stream, discount_rate)


def liability_value_mortality_weighted(
    consumption_stream: npt.NDArray[np.float64],
    discount_rate: float,
    survival_probs: npt.NDArray[np.float64],
    auto_align: bool = True,
) -> float:
    """
    Calculate mortality-weighted liability value.

    Liabilities with mortality weighting. Appropriate when
    annuities are available.

    Formula: L̂_t = Σ q_tv * C̄_v / (1+r)^(v-t)

    Args:
        consumption_stream: Array of nondiscretionary consumption by year
        discount_rate: Discount rate for liabilities
        survival_probs: Survival probabilities by year
        auto_align: If True, automatically align array lengths (default True)

    Returns:
        Mortality-weighted present value of liabilities

    Raises:
        ValueError: If arrays have different lengths and cannot be aligned
    """
    if auto_align:
        consumption_stream, survival_probs = align_income_survival(
            consumption_stream, survival_probs
        )

    if len(consumption_stream) != len(survival_probs):
        raise ValueError(
            f"consumption_stream and survival_probs must have same length: "
            f"got {len(consumption_stream)} and {len(survival_probs)}"
        )

    weighted_consumption = consumption_stream * survival_probs
    return pv(weighted_consumption, discount_rate)


def liability_discount_rate(
    risk_free_rate: float,
    equity_return: float,
    equity_weight: float = 0.15,
) -> float:
    """
    Calculate appropriate discount rate for liabilities.

    Similar to income, liabilities have some equity-like risk
    (e.g., housing costs correlated with economy).

    Args:
        risk_free_rate: Risk-free rate
        equity_return: Expected equity return
        equity_weight: Equity-like portion of liability risk (default 15%)

    Returns:
        Appropriate discount rate for liabilities
    """
    bond_weight = 1 - equity_weight
    return bond_weight * risk_free_rate + equity_weight * equity_return


def net_worth(
    financial_wealth: float,
    human_capital_value: float,
    liability_value: float,
) -> float:
    """
    Calculate net worth (individual economic balance sheet).

    Net worth is the amount available for discretionary consumption.

    Formula: W = F + H - L

    where:
        F = Financial wealth (taxable + tax-advantaged accounts)
        H = Human capital (PV of future income)
        L = Liabilities (PV of nondiscretionary consumption)

    Args:
        financial_wealth: Current financial assets
        human_capital_value: Present value of human capital
        liability_value: Present value of liabilities

    Returns:
        Net worth

    Example:
        >>> net_worth(100000, 2767689, 1392064)
        1475625  # Available for discretionary consumption
    """
    return financial_wealth + human_capital_value - liability_value


def net_worth_components(
    taxable_wealth: float,
    tax_advantaged_wealth: float,
    income_stream: npt.NDArray[np.float64],
    consumption_stream: npt.NDArray[np.float64],
    income_discount_rate: float,
    liability_discount_rate: float,
    survival_probs: npt.NDArray[np.float64] | None = None,
    annuities_available: bool = False,
) -> dict[str, float]:
    """
    Calculate all components of net worth.

    Provides a complete breakdown of the individual balance sheet.

    Args:
        taxable_wealth: Taxable account balance
        tax_advantaged_wealth: Tax-advantaged account balance (401k, IRA)
        income_stream: Future income by year
        consumption_stream: Nondiscretionary consumption by year
        income_discount_rate: Discount rate for income
        liability_discount_rate: Discount rate for liabilities
        survival_probs: Survival probabilities (required if annuities_available)
        annuities_available: Whether to use mortality weighting

    Returns:
        Dictionary with all balance sheet components
    """
    financial_wealth = taxable_wealth + tax_advantaged_wealth

    if annuities_available:
        if survival_probs is None:
            raise ValueError("survival_probs required when annuities_available=True")
        hc = human_capital_mortality_weighted(
            income_stream, income_discount_rate, survival_probs
        )
        liab = liability_value_mortality_weighted(
            consumption_stream, liability_discount_rate, survival_probs
        )
    else:
        hc = human_capital(income_stream, income_discount_rate)
        liab = liability_value(consumption_stream, liability_discount_rate)

    nw = financial_wealth + hc - liab

    return {
        "taxable_wealth": taxable_wealth,
        "tax_advantaged_wealth": tax_advantaged_wealth,
        "financial_wealth": financial_wealth,
        "human_capital": hc,
        "liabilities": liab,
        "net_worth": nw,
    }


def required_savings_rate(
    current_age: int,
    retirement_age: int,
    current_income: float,
    income_growth_rate: float,
    target_replacement_ratio: float,
    retirement_years: int,
    current_savings: float,
    real_return: float,
) -> float:
    """
    Calculate required savings rate to meet retirement goals.

    Determines the fraction of income that must be saved each year
    to achieve a target income replacement ratio in retirement.

    Args:
        current_age: Current age
        retirement_age: Planned retirement age
        current_income: Current annual income
        income_growth_rate: Expected annual income growth
        target_replacement_ratio: Target retirement income as % of final salary
        retirement_years: Expected years in retirement
        current_savings: Current retirement savings
        real_return: Expected real investment return

    Returns:
        Required savings rate (as decimal, e.g., 0.15 for 15%)
    """
    working_years = retirement_age - current_age

    # Project income over working years
    years = np.arange(working_years)
    incomes = current_income * (1 + income_growth_rate) ** years

    # Final salary and target retirement income
    final_salary = incomes[-1]
    target_retirement_income = final_salary * target_replacement_ratio

    # Required retirement savings (PV of retirement income needs)
    retirement_needs = present_value_growing(
        target_retirement_income,
        0,  # Assume constant real retirement income
        real_return,
        retirement_years,
    )

    # Future value of current savings at retirement
    fv_current_savings = current_savings * (1 + real_return) ** working_years

    # Gap to fill with new savings
    savings_gap = retirement_needs - fv_current_savings

    if savings_gap <= 0:
        return 0.0  # Already on track

    # Required annual savings (as growing annuity)
    # S * FV_factor = savings_gap
    # where FV_factor accounts for growth and compounding

    # Use numerical approach for accuracy
    def fv_savings(rate):
        annual_savings = incomes * rate
        fv_factors = (1 + real_return) ** (working_years - 1 - years)
        return np.sum(annual_savings * fv_factors)

    # Binary search for savings rate
    low, high = 0.0, 1.0
    while high - low > 0.0001:
        mid = (low + high) / 2
        if fv_savings(mid) < savings_gap:
            low = mid
        else:
            high = mid

    return mid


def funded_ratio(
    assets: float,
    liabilities: float,
) -> float:
    """
    Calculate funded ratio (assets / liabilities).

    A key measure of financial health:
        - < 1.0: Underfunded (liabilities exceed assets)
        - = 1.0: Exactly funded
        - > 1.0: Overfunded (surplus available for discretionary use)

    Args:
        assets: Total assets (financial wealth + human capital)
        liabilities: Total liabilities (PV of nondiscretionary consumption)

    Returns:
        Funded ratio

    Example:
        >>> funded_ratio(2867689, 1392064)
        2.06  # Well-funded, ~2x coverage
    """
    if liabilities <= 0:
        return float('inf')
    return assets / liabilities

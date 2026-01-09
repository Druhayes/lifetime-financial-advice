"""
Portfolio Optimization for Lifecycle Financial Planning

Implements Mean-Variance Optimization (MVO) and related functions:
    - Portfolio return and risk calculations
    - MVO with and without taxes
    - Risk tolerance to risk aversion parameter conversion
    - Asset allocation for net worth components

Based on Chapters 7, 8, and 9 of "Lifetime Financial Advice".

References:
    - Markowitz, H.M. (1952). Portfolio Selection.
    - Sharpe, W.F. (1964). Capital Asset Prices.
    - Levy, H. & Markowitz, H.M. (1979). Approximating Expected Utility.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class AssetClass:
    """Defines an asset class for portfolio optimization."""
    name: str
    expected_return: float
    std_dev: float


# Default asset classes from the document
# Based on long-term capital market assumptions
DEFAULT_ASSET_CLASSES = [
    AssetClass("US Large Cap", 0.070, 0.155),
    AssetClass("US Small Cap", 0.080, 0.200),
    AssetClass("International Developed", 0.065, 0.170),
    AssetClass("Emerging Markets", 0.085, 0.250),
    AssetClass("US Bonds", 0.035, 0.045),
    AssetClass("TIPS", 0.025, 0.055),
    AssetClass("Cash", 0.020, 0.010),
]


def portfolio_expected_return(
    weights: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Calculate expected return of a portfolio.

    Formula: E[R_p] = Σ w_i * μ_i = w' * μ

    Args:
        weights: Array of asset weights (sum to 1)
        returns: Array of expected returns for each asset

    Returns:
        Portfolio expected return

    Example:
        >>> weights = np.array([0.6, 0.4])
        >>> returns = np.array([0.07, 0.03])
        >>> portfolio_expected_return(weights, returns)
        0.054  # 5.4%
    """
    weights = np.asarray(weights)
    returns = np.asarray(returns)
    return float(np.dot(weights, returns))


def portfolio_variance(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Calculate portfolio variance.

    Formula: σ²_p = w' * V * w

    where V is the covariance matrix.

    Args:
        weights: Array of asset weights
        cov_matrix: Covariance matrix of returns

    Returns:
        Portfolio variance

    Example:
        >>> weights = np.array([0.6, 0.4])
        >>> cov = np.array([[0.04, 0.01], [0.01, 0.01]])
        >>> portfolio_variance(weights, cov)
        0.019  # 1.9%
    """
    weights = np.asarray(weights)
    cov_matrix = np.asarray(cov_matrix)
    return float(np.dot(weights, np.dot(cov_matrix, weights)))


def portfolio_std(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Calculate portfolio standard deviation.

    Args:
        weights: Array of asset weights
        cov_matrix: Covariance matrix of returns

    Returns:
        Portfolio standard deviation
    """
    return np.sqrt(portfolio_variance(weights, cov_matrix))


def covariance_matrix(
    std_devs: np.ndarray,
    correlations: np.ndarray,
) -> np.ndarray:
    """
    Build covariance matrix from standard deviations and correlations.

    Formula: Cov(i,j) = σ_i * σ_j * ρ_ij

    Args:
        std_devs: Array of asset standard deviations
        correlations: Correlation matrix (symmetric, ones on diagonal)

    Returns:
        Covariance matrix
    """
    std_devs = np.asarray(std_devs)
    correlations = np.asarray(correlations)

    n = len(std_devs)
    cov = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cov[i, j] = std_devs[i] * std_devs[j] * correlations[i, j]

    return cov


def levy_markowitz_utility(
    weights: np.ndarray,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    theta: float,
) -> float:
    """
    Calculate Levy-Markowitz utility for a portfolio.

    Approximates expected utility using mean and variance.

    Formula: U ≈ ((1+μ)^θ - 1)/θ + 0.5*(θ-1)*(1+μ)^(θ-2) * σ²

    Args:
        weights: Portfolio weights
        returns: Expected returns
        cov_matrix: Covariance matrix
        theta: Risk tolerance parameter

    Returns:
        Levy-Markowitz utility
    """
    mu = portfolio_expected_return(weights, returns)
    var = portfolio_variance(weights, cov_matrix)

    one_plus_mu = 1 + mu

    if np.isclose(theta, 1.0):
        utility = np.log(one_plus_mu)
    else:
        utility = (one_plus_mu ** theta - 1) / theta

    # Variance penalty
    if np.isclose(theta, 1.0):
        second_deriv = -1 / one_plus_mu ** 2
    else:
        second_deriv = (theta - 1) * one_plus_mu ** (theta - 2)

    utility += 0.5 * second_deriv * var

    return float(utility)


def risk_adjusted_expected_return(
    weights: np.ndarray,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    lambda_: float,
) -> float:
    """
    Calculate Risk-Adjusted Expected Return (RAER).

    Formula: RAER = μ - (λ/2) * σ²

    Args:
        weights: Portfolio weights
        returns: Expected returns
        cov_matrix: Covariance matrix
        lambda_: Risk aversion parameter

    Returns:
        Risk-adjusted expected return
    """
    mu = portfolio_expected_return(weights, returns)
    var = portfolio_variance(weights, cov_matrix)
    return mu - (lambda_ / 2) * var


def mvo_optimal_portfolio(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    theta: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    use_raer: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Find optimal portfolio using Mean-Variance Optimization.

    Maximizes Levy-Markowitz utility (or RAER) subject to constraints.

    Args:
        returns: Expected returns for each asset class
        cov_matrix: Covariance matrix
        theta: Risk tolerance parameter
        min_weight: Minimum weight for any asset (default 0)
        max_weight: Maximum weight for any asset (default 1)
        use_raer: If True, use RAER instead of Levy-Markowitz

    Returns:
        Tuple of (optimal_weights, expected_return, std_dev)

    Example:
        >>> returns = np.array([0.07, 0.035])
        >>> cov = np.array([[0.0225, 0.002], [0.002, 0.002]])
        >>> weights, ret, std = mvo_optimal_portfolio(returns, cov, 0.35)
        >>> weights
        array([0.25, 0.75])  # 25% equity, 75% bonds
    """
    n_assets = len(returns)

    # Convert theta to lambda if using RAER
    lambda_ = theta_to_lambda(theta, returns, cov_matrix) if use_raer else None

    def objective(w):
        if use_raer:
            return -risk_adjusted_expected_return(w, returns, cov_matrix, lambda_)
        else:
            return -levy_markowitz_utility(w, returns, cov_matrix, theta)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]

    # Bounds
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    # Initial guess (equal weight)
    x0 = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x
    expected_ret = portfolio_expected_return(optimal_weights, returns)
    std_dev = portfolio_std(optimal_weights, cov_matrix)

    return optimal_weights, expected_ret, std_dev


def efficient_frontier(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the efficient frontier.

    Returns portfolios that maximize return for each risk level.

    Args:
        returns: Expected returns
        cov_matrix: Covariance matrix
        n_points: Number of points on frontier
        min_weight: Minimum asset weight
        max_weight: Maximum asset weight

    Returns:
        Tuple of (weights_array, returns_array, std_devs_array)
    """
    n_assets = len(returns)

    # Find min and max return portfolios
    min_ret = np.min(returns)
    max_ret = np.max(returns)

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_weights = []
    frontier_returns = []
    frontier_stds = []

    for target in target_returns:
        # Minimize variance for given return target
        def objective(w):
            return portfolio_variance(w, cov_matrix)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: portfolio_expected_return(w, returns) - target},
        ]

        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                frontier_weights.append(result.x)
                frontier_returns.append(target)
                frontier_stds.append(np.sqrt(result.fun))
        except Exception:
            continue

    return (
        np.array(frontier_weights),
        np.array(frontier_returns),
        np.array(frontier_stds),
    )


def theta_to_lambda(
    theta: float,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Convert risk tolerance (theta) to risk aversion (lambda).

    Finds the lambda value such that the optimal portfolio under RAER
    matches the optimal portfolio under Levy-Markowitz with theta.

    Args:
        theta: Risk tolerance parameter
        returns: Expected returns
        cov_matrix: Covariance matrix

    Returns:
        Corresponding risk aversion parameter lambda

    Example:
        >>> theta_to_lambda(0.35, returns, cov)
        2.72
    """
    # Find optimal portfolio with theta
    weights_lm, _, _ = mvo_optimal_portfolio(
        returns, cov_matrix, theta, use_raer=False
    )

    # For this portfolio, find lambda
    mu = portfolio_expected_return(weights_lm, returns)
    var = portfolio_variance(weights_lm, cov_matrix)

    if var <= 0:
        return 0.0

    # From first-order conditions of RAER maximization
    # At optimum: μ_i - λ * Σ_j V_ij * w_j = constant
    # This gives us lambda = 2 * (μ - rf) / σ² approximately

    # More precise: use the marginal utility relationship
    one_plus_mu = 1 + mu

    # d(LM)/dσ² = 0.5 * u''(1+μ) = 0.5 * (θ-1) * (1+μ)^(θ-2)
    # d(RAER)/dσ² = -λ/2
    # Setting equal: λ = -(θ-1) * (1+μ)^(θ-2)

    lambda_ = -(theta - 1) * one_plus_mu ** (theta - 2)

    return max(0, lambda_)


def certainty_equivalent_return(
    expected_return: float,
    volatility: float,
    theta: float,
) -> float:
    """
    Calculate certainty equivalent return.

    The constant return that provides the same utility as the risky return.

    Formula: h = μ - (1-θ)/(2θ) * σ_log²

    where σ_log is the log-return volatility.

    Args:
        expected_return: Expected return
        volatility: Standard deviation of returns
        theta: Risk tolerance parameter

    Returns:
        Certainty equivalent return

    Example:
        >>> certainty_equivalent_return(0.0333, 0.10, 0.35)
        0.0291  # 2.91% vs 3.33% expected
    """
    # Convert to log volatility (approximately equal for small vol)
    log_vol = volatility

    if theta <= 0:
        raise ValueError("theta must be positive")

    ce = expected_return - ((1 - theta) / (2 * theta)) * log_vol ** 2
    return ce


def human_capital_allocation(
    equity_correlation: float = 0.2,
) -> dict:
    """
    Model human capital as an asset class portfolio.

    Human capital has both bond-like and equity-like characteristics.
    The equity-like component reflects correlation with economic cycles.

    Args:
        equity_correlation: Correlation of income with equity returns

    Returns:
        Dictionary with equity and bond weights

    Example:
        >>> human_capital_allocation(0.2)
        {"equity": 0.2, "bond": 0.8}  # 20% equity-like
    """
    return {"equity": equity_correlation, "bond": 1 - equity_correlation}


def liability_allocation(
    equity_correlation: float = 0.15,
) -> dict:
    """
    Model liabilities as an asset class portfolio.

    Liabilities (e.g., housing costs) may have some correlation
    with economic conditions.

    Args:
        equity_correlation: Correlation of liabilities with equity

    Returns:
        Dictionary with equity and bond weights
    """
    return {"equity": equity_correlation, "bond": 1 - equity_correlation}


def net_worth_allocation(
    financial_wealth: float,
    taxable_weights: np.ndarray,
    tax_advantaged_weights: np.ndarray,
    human_capital: float,
    hc_equity_weight: float,
    liabilities: float,
    liability_equity_weight: float,
) -> dict:
    """
    Calculate implied allocation of total net worth.

    Combines:
        - Financial wealth (with explicit asset allocation)
        - Human capital (modeled as stock/bond mix)
        - Liabilities (modeled as stock/bond mix)

    Args:
        financial_wealth: Total financial assets
        taxable_weights: Weights in taxable account
        tax_advantaged_weights: Weights in tax-advantaged account
        human_capital: Value of human capital
        hc_equity_weight: Equity weight of human capital
        liabilities: Value of liabilities
        liability_equity_weight: Equity weight of liabilities

    Returns:
        Dictionary with net worth equity and bond weights
    """
    # Total assets
    total_assets = financial_wealth + human_capital

    # Equity in financial wealth (simplified to total equity)
    # Assuming first half of weights are equity classes
    n_assets = len(taxable_weights)
    n_equity = n_assets // 2
    fin_equity = financial_wealth * (
        np.sum(taxable_weights[:n_equity]) + np.sum(tax_advantaged_weights[:n_equity])
    ) / 2

    # Equity in human capital
    hc_equity = human_capital * hc_equity_weight

    # Equity in liabilities
    liab_equity = liabilities * liability_equity_weight

    # Net worth
    net_worth = total_assets - liabilities

    # Net equity exposure
    net_equity = fin_equity + hc_equity - liab_equity

    equity_weight = net_equity / net_worth if net_worth > 0 else 0

    return {
        "equity_weight": equity_weight,
        "bond_weight": 1 - equity_weight,
        "net_worth": net_worth,
    }


def reverse_optimization(
    reference_weights: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """
    Perform reverse optimization to infer expected returns.

    Given a reference portfolio (e.g., market portfolio), infer the
    expected returns that would make it optimal.

    Formula: μ = λ * V * w + r_f

    Args:
        reference_weights: Weights of reference portfolio
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate

    Returns:
        Implied expected returns

    Note:
        This requires knowing the market's risk aversion λ.
        We use λ that makes the reference optimal.
    """
    # This is a simplified implementation
    # Full version requires iterative solve for lambda

    # Assume lambda such that reference is optimal
    # This is often calibrated to give reasonable equity premium

    reference_weights = np.asarray(reference_weights)
    cov_matrix = np.asarray(cov_matrix)

    # Use lambda = 2.5 as starting point (typical market risk aversion)
    lambda_ = 2.5

    # Implied returns: μ = r_f + λ * V * w
    implied_returns = risk_free_rate + lambda_ * np.dot(cov_matrix, reference_weights)

    return implied_returns


def black_litterman(
    reference_weights: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    views: List[Dict],
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement Black-Litterman model for incorporating views.

    Combines equilibrium returns with investor views to produce
    posterior expected returns.

    Args:
        reference_weights: Market/reference portfolio weights
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        views: List of views, each with 'assets', 'return', 'confidence'
        tau: Uncertainty in equilibrium returns

    Returns:
        Tuple of (posterior_returns, posterior_weights)

    Example:
        >>> views = [
        ...     {"assets": [0], "return": 0.08, "confidence": 0.5},  # Bullish on asset 0
        ... ]
        >>> post_ret, post_wts = black_litterman(mkt_wts, cov, 0.02, views)
    """
    # Get equilibrium returns
    pi = reverse_optimization(reference_weights, cov_matrix, risk_free_rate)

    n_assets = len(reference_weights)
    n_views = len(views)

    if n_views == 0:
        return pi, reference_weights

    # Build view matrices
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = np.zeros(n_views)

    for i, view in enumerate(views):
        for asset_idx in view["assets"]:
            P[i, asset_idx] = 1.0 / len(view["assets"])
        Q[i] = view["return"]
        omega_diag[i] = (1 - view["confidence"]) / view["confidence"] * tau

    Omega = np.diag(omega_diag)

    # Posterior returns
    tau_cov = tau * cov_matrix
    term1 = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
    term2 = np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q

    posterior_returns = term1 @ term2

    # Posterior optimal weights (using same lambda)
    weights, _, _ = mvo_optimal_portfolio(
        posterior_returns, cov_matrix, 0.566  # Reference theta
    )

    return posterior_returns, weights

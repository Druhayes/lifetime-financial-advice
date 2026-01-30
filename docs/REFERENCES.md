# Source Material References

## Primary Source

**Idzorek, T.M., & Kaplan, P.D. (2024)**
*"Lifetime Financial Advice: A Personalized Optimal Multilevel Approach"*
CFA Institute Research Foundation
With a Foreword by Roger G. Ibbotson

Available in MCP Knowledge Database: `lifetime-financial-advice.pdf`

## Important Note on Source Evolution

This book builds upon and supersedes the earlier work:

**Ibbotson, R.G., Milevsky, M.A., Chen, P., & Zhu, K.X. (2007)**
*"Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"*
Research Foundation of CFA Institute

The 2024 version by Idzorek and Kaplan provides a "personalized optimal multilevel approach"
with a three-stage model:
1. **Parent Model**: Life-cycle utility maximization
2. **Child Model**: Net-worth asset allocation and location optimization
3. **Grandchild Model**: Alpha-tracking error optimization for implementation

## Chapter-by-Chapter Implementation

### Part I: Parent Life-Cycle Model

#### Chapter 3: Utility Theory and Preferences (Pages 52-73)
**Implemented in:** `utility.py`
- CRRA utility function (Equation 8.2, Page 153)
- Risk tolerance parameter (θ or theta)
- Investor preference parameters:
  - ρ (rho): Impatience for consumption
  - η (eta): Elasticity of intertemporal substitution (EOIS)
  - γ (gamma): Intergenerational elasticity (bequest flexibility)
  - φ (phi): Bequest motive strength

**Key Functions:**
- `crra_utility()` - CRRA utility with theta parameter
- `levy_markowitz_utility()` - Mean-variance approximation (Equation 8.1)

#### Appendix 3A: Gompertz Survival Formula (Pages 77-78)
**Implemented in:** `mortality.py`
- Truncated Gompertz function for survival probability
- Gender-specific parameters (Exhibit 3A.1):
  - Men: M = 88, B = 10.65
  - Women: M = 91, B = 8.88
- Joint survivor probability for couples

**Key Functions:**
- `gompertz_survival_prob()` - Core Gompertz formula
- `survival_probability()` - Single person survival
- `joint_survival_probability()` - Couples

#### Chapter 4: Human Capital (Pages 80-90)
**Implemented in:** `present_value.py`, `balance_sheet.py`
- Individual balance sheet framework (Page 81):
  - Assets = Financial Wealth + Human Capital
  - Liabilities = PV of nondiscretionary consumption
  - Net Worth = Assets - Liabilities
- Human capital present value (Equations 4.6-4.8, Pages 89-90)
- Discount rate selection (Page 86)
- Mortality weighting (Pages 85-86)

**Key Functions:**
- `present_value_human_capital()` - HC valuation with mortality
- `IndividualBalanceSheet` - Complete balance sheet framework

#### Chapter 5: Deterministic Life-Cycle Models (Pages 98-109)
**Implemented in:** `spending.py`
- Optimal consumption timing
- Consumption growth rate: g = (r - ρ)/(1 - η)
- Deterministic divisor calculations

**Key Functions:**
- `optimal_consumption_growth_rate()` - Intertemporal consumption

#### Chapter 6: Stochastic Life-Cycle Models (Pages 110-132)
**Implemented in:** `spending.py`
- Stochastic consumption rules
- Consumption divisor (Equations 6.33-6.34, Pages 131-132)
- Certainty equivalent return (Page 122)
- Annuity effects on consumption

**Key Functions:**
- Consumption divisor calculations with market risk

### Part II: Child Net-Worth Optimization Model

#### Chapter 8: Mean-Variance Optimization (Pages 153-171)
**Implemented in:** `utility.py`, `portfolio.py`
- Levy-Markowitz utility function (Equation 8.1, Page 153)
- CRRA utility for MVO (Equation 8.2, Page 153)
- Expected utility maximization in mean-variance framework
- Risk tolerance (θ) parameter linking to life-cycle models

**Key Equations:**
- Equation 8.1: E[u(1+R)] ≈ u(1+μ) + 0.5 * u''(1+μ) * σ²
- Equation 8.2: u(x) = ln(x) if θ = 1, else (x^θ - 1)/θ
- Equation 8.3: u''(x) = (θ-1) * x^(θ-2)

**Key Functions:**
- `levy_markowitz_utility()` - Links life-cycle to MVO
- `mvo_optimal_portfolio()` - Portfolio optimization

### Part III: Grandchild Alpha-Tracking Error Optimization

#### Chapter 10: Black-Litterman (Advanced)
**Implementation Status:** Not verified in this alignment
**Note:** Marked as extension beyond core methodology

#### Chapter 11: Monte Carlo Simulation (Advanced)
**Implementation Status:** Not verified in this alignment
**Note:** Marked as extension beyond core methodology

### Isabela Example

#### Chapter 12: End-to-End Example (Pages 224-237)
**Implemented in:** `examples/profiles.py`, `balance_sheet.py`
- Complete financial planning example for Isabela (25-year-old female)
- Exhibits 12.1-12.20 demonstrate full three-stage model
- Parameters:
  - Age: 25
  - Salary: $75,000
  - Human Capital: $2,767,689
  - Risk Tolerance: 35%
  - EOIS: 50%
  - Impatience: 2%
  - Life Expectancy: 94 (personalized)

**Key Functions:**
- `create_isabela()` - Isabela example profile

---

## Related Academic References

- **Levy, H. & Markowitz, H.M. (1979).** "Approximating Expected Utility by a Function of Mean and Variance." *American Economic Review* 69(3): 308-317.
- **Merton, R.C. (1969).** "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case." *Review of Economics and Statistics* 51(3): 247-257.
- **Merton, R.C. (1971).** "Optimum Consumption and Portfolio Rules in a Continuous-Time Model." *Journal of Economic Theory* 3(4): 373-413.
- **Von Neumann, J. & Morgenstern, O. ([1944] 1967).** *Theory of Games and Economic Behavior*. 3rd ed. Princeton University Press.
- **Blanchett, D.M. & Kaplan, P.D. (2013).** "Alpha, Beta, and Now… Gamma." *Journal of Retirement* 1(2): 29-45.
- **Milevsky, M.A. (2020).** *Retirement Income Recipes in R*. Cambridge University Press.
- **Milevsky, M.A. (2012).** *The 7 Most Important Equations for Your Retirement*. Wiley.

---

## Verification Methodology

All formulas and parameters in this library have been verified against the 2024 source material using:
- **MCP Knowledge RAG queries** against `lifetime-financial-advice.pdf`
- **Page-specific verification** for all core formulas
- **Parameter validation** against published exhibits
- **Isabela example validation** against Chapter 12 values

**Verification Date:** 2026-01-29
**Verification Status:** All core formulas VERIFIED ✓

See `docs/source_mapping.md` for detailed function-to-page mappings.

---

## Citation

If you use this library in academic work, please cite both the software and the source material:

```bibtex
@software{lifecycle_planning,
  title = {Lifecycle Planning: A Python Library for Lifecycle Financial Planning},
  author = {Hayes, Drew},
  year = {2026},
  url = {https://github.com/Druhayes/lifetime-financial-advice}
}

@book{idzorek2024lifetime,
  title = {Lifetime Financial Advice: A Personalized Optimal Multilevel Approach},
  author = {Idzorek, Thomas M. and Kaplan, Paul D.},
  year = {2024},
  publisher = {CFA Institute Research Foundation},
  note = {With a Foreword by Roger G. Ibbotson}
}
```

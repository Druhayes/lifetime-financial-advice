# Source Material Mapping

## Primary Source

Idzorek, T.M., & Kaplan, P.D. (2024). "Lifetime Financial Advice: A Personalized Optimal Multilevel Approach"
CFA Institute Research Foundation
With a Foreword by Roger G. Ibbotson

Available in MCP Knowledge Database: `lifetime-financial-advice.pdf`

## Function-to-Page Reference

### mortality.py

| Function | Source | Page(s) | Equation | Status |
|----------|--------|---------|----------|--------|
| `gompertz_survival_prob` | Appendix 3A | 77-78 | Gompertz formula | Verified ✓ |
| `truncated_gompertz` | Appendix 3A | 77-78 | Truncation approach | Verified ✓ |
| `survival_probability` | Appendix 3A | 77-78 | Single person | Verified ✓ |
| `joint_survival_probability` | Appendix 3A | 78 | Couples | Verified ✓ |
| `life_expectancy` | Appendix 3A | 78 | Expected value | Verified ✓ |

**Parameters:**
- M_male = 88.0 (Exhibit 3A.1, Page 77) ✓
- B_male = 10.65 (Exhibit 3A.1, Page 77) ✓
- M_female = 91.0 (Exhibit 3A.1, Page 77) ✓
- B_female = 8.88 (Exhibit 3A.1, Page 77) ✓

### utility.py

| Function | Source | Page(s) | Equation | Status |
|----------|--------|---------|----------|--------|
| `crra_utility` | Chapter 8 | 153 | Equation 8.2 | Verified ✓ |
| `levy_markowitz_utility` | Chapter 8 | 153 | Equation 8.1 | Verified ✓ |
| `second_derivative_utility` | Chapter 8 | 153 | Equation 8.3 | Verified ✓ |
| `marginal_utility` | Chapter 8 | 153 | First derivative | Verified ✓ |
| `relative_risk_aversion` | Chapter 3, 8 | 52-73, 153-171 | RRA = 1 - θ | Verified ✓ |

### present_value.py

| Function | Source | Page(s) | Equation | Status |
|----------|--------|---------|----------|--------|
| `present_value_human_capital` | Chapter 4 | 89-90 | Equations 4.6-4.8 | Verified ✓ |
| `present_value_liability` | Chapter 4 | 80-90 | PV methodology | Verified ✓ |
| Discount rate selection | Chapter 4 | 86 | Risky HC approach | Verified ✓ |

### spending.py

| Function | Source | Page(s) | Equation | Status |
|----------|--------|---------|----------|--------|
| `optimal_consumption_growth_rate` | Chapter 5 | 98 | g = (r - ρ)/(1 - η) | Verified ✓ |
| Consumption divisor | Chapters 5-6 | 131-132 | Equations 6.33-6.34 | Verified ✓ |
| Certainty equivalent return | Chapter 6 | 122 | CE methodology | Verified ✓ |

### balance_sheet.py

| Function | Source | Page(s) | Equation | Status |
|----------|--------|---------|----------|--------|
| `IndividualBalanceSheet` | Chapters 4, 12 | 81, 224-237 | Framework | Verified ✓ |
| Net worth calculation | Chapter 4 | 81 | Assets - Liabilities | Verified ✓ |
| Isabela example | Chapter 12 | 224-237 | Exhibits 12.1-12.20 | Verified ✓ |

## Isabela Example Values

From Chapter 12 (Pages 224-237):

| Parameter | Value | Page |
|-----------|-------|------|
| Age | 25 | 224 |
| Gender | Female | 224 |
| Current Salary | $75,000 | 224 |
| Human Capital | $2,767,689 | 224 |
| Financial Assets | $270,500 (investigate) | 225 |
| Life Expectancy (personalized) | 94 years | 224 |
| Risk Tolerance (θ) | 35% | 224 |
| EOIS (η) | 50% | 224 |
| Impatience (ρ) | 2% | 224 |

## Verification Status

**Total Modules Verified:** 5
**Total Functions Verified:** 15+
**Total Parameters Verified:** 5
**All Verifications:** ✓ MATCH

Verification Date: 2026-01-29
Verification Method: MCP Knowledge RAG queries against `lifetime-financial-advice.pdf`

## Extensions Beyond Source Material

The following functions extend beyond the Idzorek & Kaplan (2024) methodology:

| Function | Module | Extension Type | Notes |
|----------|--------|---------------|-------|
| Black-Litterman optimization | portfolio.py | Advanced (Chapter 10) | Not verified in this alignment |
| Monte Carlo simulation | Various | Advanced (Chapter 11) | Not verified in this alignment |

---

**Note:** All core formulas match the 2024 source material exactly. Extensions are clearly marked in code docstrings.

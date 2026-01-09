# Lifetime Financial Advice Application

## Overview

This application implements a multi-level life-cycle optimization model for personalized financial advice based on the book "Lifetime Financial Advice: A Personalized Multi-Level Optimization Approach" by Idzorek and Kaplan (CFA Institute Research Foundation).

The model solves for optimal allocation to risky assets by considering an investor's complete economic balance sheet, including human capital (present value of future earnings) and liabilities (present value of future nondiscretionary consumption).

## Application Architecture

The application uses a three-level optimization framework:

1. **Parent Model**: Life-cycle utility maximization model
2. **Child Model**: Net-worth asset allocation and location optimization
3. **Grandchild Model**: Alpha-tracking error optimization for implementation

---

## User Input Parameters

### 1. Demographic Information

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `age` | int | Current age of investor | 25-80 |
| `gender` | enum | Gender for mortality/wage modeling | "male", "female" |
| `education_level` | enum | Highest education attained | "generic", "high_school", "college", "post_college" |
| `retirement_age` | int | Expected retirement age | 55-75 |
| `marital_status` | enum | For joint survival calculations | "single", "married" |
| `spouse_age` | int | Spouse's age (if married) | 25-80 |
| `spouse_gender` | enum | Spouse's gender | "male", "female" |

### 2. Financial Preference Parameters (5 Key Parameters)

These parameters drive the life-cycle optimization model:

| Greek Symbol | Parameter | Description | Range | Example |
|--------------|-----------|-------------|-------|---------|
| ρ (rho) | `subjective_discount_rate` | Impatience for consumption - preference for consuming now vs later | 0-10% | 2% (patient) |
| η (eta) | `eois` | Elasticity of Intertemporal Substitution - preference for smooth consumption | 0-100% | 50% (moderate) |
| θ (theta) | `risk_tolerance` | Investor's attitude toward risk applied to net worth | 0-100% | 35% (low-moderate) |
| γ (gamma) | `intergenerational_elasticity` | Flexibility between consumption and bequest | 0-100% | 25% (low flexibility) |
| φ (phi) | `bequest_motive_strength` | Importance of leaving a bequest vs consumption | 0-100% | 1.5% (moderate) |

#### Parameter Interpretations:

**Subjective Discount Rate (ρ):**
- Lower values (1-2%): Patient - willing to delay consumption for higher future consumption
- Higher values (5-10%): Impatient - strong preference for immediate consumption

**EOIS (η):**
- Lower values (10-30%): Strong preference for smooth/stable consumption over time
- Higher values (60-100%): Flexible - willing to accept variable consumption patterns

**Risk Tolerance (θ):**
- 0-20%: Very conservative
- 20-40%: Conservative
- 40-60%: Moderate
- 60-80%: Aggressive
- 80-100%: Very aggressive

**Bequest Parameters (γ and φ):**
- Higher γ: More sensitive bequest size to changes in bequest motive strength
- Higher φ: Larger bequest at expense of consumption

### 3. Current Financial Position

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_salary` | float | Current annual income (for wage projection scaling) |
| `taxable_account_balance` | float | Value of taxable investment accounts |
| `tax_deferred_account_balance` | float | Value of 401(k), traditional IRA, etc. |
| `tax_exempt_account_balance` | float | Value of Roth accounts |
| `employer_match_rate` | float | Employer 401(k) match percentage |
| `employer_match_limit` | float | Maximum employer match dollar amount |

### 4. Income Sources

| Parameter | Type | Description |
|-----------|------|-------------|
| `has_defined_benefit_pension` | bool | Whether investor has DB pension |
| `pension_annual_amount` | float | Expected annual pension at retirement |
| `expected_social_security` | float | Expected annual Social Security benefit |
| `other_guaranteed_income` | float | Annuities, rental income, etc. |

### 5. Consumption & Liabilities

| Parameter | Type | Description |
|-----------|------|-------------|
| `annual_nondiscretionary_consumption` | float | Essential spending (food, housing, utilities) |
| `mortgage_balance` | float | Outstanding mortgage debt |
| `other_debt` | float | Other outstanding debts |
| `desired_bequest_amount` | float | Target bequest in real dollars (if applicable) |

### 6. Longevity Personalization

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `life_expectancy_override` | int | Custom life expectancy based on family history | Auto-calculated |
| `family_longevity` | enum | General family longevity pattern | "average", "below_average", "above_average" |

---

## Human Capital Model

Human capital is modeled as the present value of future earnings, treated as an implicit asset on the balance sheet.

### Wage Projection Model

Wages are projected using age-gender-education curves and scaled to the investor's actual current salary:

```
salary_multiplier = current_salary / baseline_salary(age, gender, education)
projected_salary(future_age) = baseline_salary(future_age, gender, education) * salary_multiplier
```

### Human Capital Risk Characteristics

Human capital is modeled as having asset-class-like risk characteristics:

| Investor Type | Stock Allocation | Bond Allocation | Description |
|---------------|------------------|-----------------|-------------|
| Stockbroker | 45% | 55% | Income tied to market |
| Typical (Isabela) | 20% | 80% | Moderate income stability |
| Tenured Professor | 10% | 90% | Very stable income |

**Parameters for human capital modeling:**
- `human_capital_stock_weight`: 0-100% (default based on occupation type)
- `occupation_stability`: "stable", "moderate", "volatile"

### Human Capital Discount Rate

The discount rate for human capital equals the expected return of the representative stock/bond mix:

```
k_y = stock_weight * E[R_stock] + bond_weight * E[R_bond]
```

---

## Longevity Model (Gompertz)

The Gompertz model calculates survival probabilities using three parameters:

### Base Parameters (US Mortality Rates)

| Parameter | Men | Women |
|-----------|-----|-------|
| Mode (m) | 88 | 91 |
| Dispersion (b) | 10.65 | 8.88 |

### Gompertz Survival Probability Formula

```
P(survive to age a2 | alive at age a1) = exp(-exp((a1-m)/b) * (exp((a2-a1)/b) - 1))
```

### Personalization

Life expectancy can be adjusted based on:
- Family health history
- Personal health conditions
- Lifestyle factors

Default life expectancy (25-year-old woman): 86.4 years

---

## Capital Market Assumptions

### Three-Asset Class Model (for Human Capital)

| Asset Class | Expected Return | Standard Deviation |
|-------------|-----------------|-------------------|
| Domestic Stocks | 4.72% | 15.88% |
| International Stocks | 5.04% | 17.18% |
| Bonds | 2.75% | 5.62% |

### Correlations

| | Domestic Stocks | International Stocks | Bonds |
|---|-----------------|---------------------|-------|
| Domestic Stocks | 1.00 | 0.87 | 0.21 |
| International Stocks | 0.87 | 1.00 | 0.37 |
| Bonds | 0.21 | 0.37 | 1.00 |

### Ten-Asset Class Model (for Portfolio Optimization)

| Asset Class | Expected Return | Standard Deviation | Beta |
|-------------|-----------------|-------------------|------|
| US Large-Cap Stocks | 4.68% | 15.42% | 1.43 |
| US Mid/Small-Cap Stocks | 5.01% | 17.95% | 1.65 |
| Global DM ex-US Stocks | 5.05% | 16.71% | 1.67 |
| Emerging Market Stocks | 5.40% | 21.42% | 1.91 |
| US Bonds | 2.69% | 3.79% | 0.12 |
| Inflation-Linked Bonds | 2.88% | 5.81% | 0.24 |
| Municipal Bonds | N/A | N/A | 0.14 |
| Global Bonds ex-US | 3.29% | 8.33% | 0.51 |
| Cash | 2.50% | 0.55% | 0.00 |

**Risk-Free Rate:** 2.50%

---

## Balance Sheet Framework

### Assets (Left Side)

1. **Financial Wealth (F)**
   - Taxable accounts
   - Tax-deferred accounts (401k, Traditional IRA)
   - Tax-exempt accounts (Roth)

2. **Human Capital (H)**
   - Present value of wage income
   - Present value of defined benefit pension
   - Present value of Social Security

### Liabilities (Right Side)

1. **Consumption-Related Liabilities (L_c)**
   - Present value of nondiscretionary consumption

2. **Life Insurance-Related Liabilities (L_i)**
   - Present value of term life premiums for bequest

### Net Worth

```
Net Worth (W) = Financial Wealth (F) + Human Capital (H) - Liabilities (L)
```

---

## Optimization Framework

### Utility Functions

**CRRA (Constant Relative Risk Aversion) Utility:**
```
u_θ(x) = ln(x)           if θ = 1
u_θ(x) = (x^θ - 1) / θ   if θ ≠ 1
```

**Levy-Markowitz Utility Approximation:**
```
EU ≈ u(1 + μ) + (1/2) * u''(1 + μ) * σ²
```

### Mean-Variance Optimization (MVO)

The optimization maximizes the Levy-Markowitz utility function:

```
max: μ_portfolio - (1/(2θ)) * σ²_portfolio
subject to: Σ weights = 1, weights ≥ 0
```

### Net-Worth Optimization

Risk tolerance applies to net worth, not just financial assets:

```
R_W = (F'/W) * R_F + (H'/W) * R_H - (L'/W) * R_L
```

Where primes (') indicate cash-flow adjusted values.

### Risk Aversion Parameter Mapping

```
λ (risk aversion) = 1 / θ (risk tolerance)
```

For θ = 35%, λ ≈ 2.72

---

## Outputs

### Primary Outputs

1. **Optimal Asset Allocation**
   - Target allocation for taxable accounts
   - Target allocation for tax-deferred accounts
   - Target allocation for tax-exempt accounts
   - Net-worth implied allocation

2. **Lifetime Financial Plan**
   - Optimal saving rate
   - Optimal spending schedule (nondiscretionary + discretionary)
   - Optimal bequest size

3. **Insurance Recommendations**
   - Term life insurance amount needed
   - Annuity allocation at retirement

### Supporting Outputs

- Human capital value estimate
- Liability value estimate
- Net worth estimate
- Probability distributions for future outcomes (Monte Carlo)

---

## Example: "Isabela" Case Study

| Parameter | Value |
|-----------|-------|
| Age | 25 |
| Gender | Female |
| Education | Post-college (Master's) |
| Current Salary | $75,000 |
| Retirement Age | 65 |
| Subjective Discount Rate (ρ) | 2% |
| EOIS (η) | 50% |
| Risk Tolerance (θ) | 35% |
| Intergenerational Elasticity (γ) | 25% |
| Bequest Motive Strength (φ) | 1.5% |
| Nondiscretionary Consumption | $40,000/year |
| Target Bequest | $1,000,000 |
| Life Expectancy Override | 94 (family history) |

**Resulting Allocation:**
- Financial Assets: ~86.5% stocks / 13.5% bonds
- Net Worth: ~35% stocks / 65% bonds

---

## Implementation Notes

### Data Sources

- Wage curves: US Bureau of Labor Statistics
- Mortality tables: US Social Security Administration
- Capital market assumptions: Historical data + forward-looking adjustments

### Key Dependencies

- Numerical optimization library (scipy.optimize or cvxpy)
- Monte Carlo simulation capability
- Present value calculations
- Gompertz survival probability functions

### Calculation Flow

1. Collect user inputs (demographics, preferences, finances)
2. Calculate human capital (project wages, discount to present value)
3. Calculate liabilities (project consumption, discount to present value)
4. Calculate net worth
5. Run MVO optimization with net-worth framework
6. Generate asset location recommendations
7. Project lifetime consumption and bequest paths
8. Run Monte Carlo for probability distributions

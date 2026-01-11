# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-11

### Added

#### Core Library
- Gompertz mortality model for survival probability calculations
- CRRA utility functions and preference modeling
- Human capital valuation with mortality weighting
- Present value calculations for assets and liabilities
- Income projection with age-gender-education wage curves
- Optimal consumption and spending rules
- Mean-variance portfolio optimization
- SPIA and annuity pricing with mortality credits
- Individual economic balance sheet framework

#### Package Structure
- Converted to PyPI-ready package with src layout
- Package name: `lifecycle-planning` (import as `lifecycle_planning`)
- Organized into core modules: mortality, utility, present_value, income, spending, portfolio, annuities, balance_sheet

#### Examples
- Isabela case study: comprehensive 25-year-old professional example
- Quick demo showcasing all library features
- Example profiles: create_isabela(), create_conservative_retiree(), create_aggressive_professional()

#### CLI
- `lifecycle` command-line interface
- `lifecycle --version`: Show package version
- `lifecycle demo`: Run quick demonstration
- `lifecycle demo isabela`: Run Isabela case study
- `lifecycle info`: Display package information

#### Documentation
- Comprehensive README with usage examples
- Installation instructions for both repo and PyPI
- Parameter guidance for different client types
- Mathematical model descriptions

#### Development
- Type hints throughout codebase
- pytest test infrastructure
- Code formatting with black
- Linting with ruff
- Type checking with mypy

### Changed
- Migrated from `lifecycle` to `lifecycle_planning` package namespace
- Updated all imports to use new package structure
- Reorganized examples into standalone scripts

### Technical Details
- Python 3.11+ required
- Dependencies: numpy>=1.24.0, scipy>=1.10.0
- Optional dependencies for CLI, docs, and development
- Build backend: hatchling
- License: MIT

[0.1.0]: https://github.com/Druhayes/lifetime-financial-advice/releases/tag/v0.1.0

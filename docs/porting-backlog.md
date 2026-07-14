# Porting backlog — Engine A capabilities → `core/domain/`

`core/models/` (Engine A) is a **frozen executable spec**: hand-written OOP retirement
modelling, exercised only by `notebooks/`, with capabilities the live engine
(`core/domain/`, wired to the app via `core.domain.build_scenario`) does not have yet.
Do not extend `core/models/` and do not delete it wholesale — each capability below is
ported as its own piece of feature work, and a models file is deleted only when every
capability it holds has landed in `core/domain/`.

Porting discipline (same rules as the 2026-07 core/ restructuring):

1. **Characterize first.** Pin the models/notebook behaviour with pytest + goldens
   before writing the domain version; the notebook is the reference run.
2. **Domain stays pure.** No firebase/yfinance/cpi/flask/disk imports — data comes in
   as arguments (see `core/domain/returns.py` for the seam pattern). I/O adapters go
   in `core/services/`.
3. **Refresh constants.** Engine A is pinned to tax year 2024 (SS wage cap $168,600,
   2024 MFJ brackets); port the logic, not the constants — `core/domain/` carries 2025
   values.
4. **Update the notebook** that exercised the capability to import from `core/domain/`
   once ported, then delete the ported code from `core/models/` (file-by-file, per the
   deletion map at the bottom).

## Capabilities

### Drawdown

| Capability | Target | Source |
|---|---|---|
| Per-lot cost-basis withdrawals (track basis per holding; capital gains computed per sale instead of one portfolio-wide `gain_fraction` scalar) | `core/domain/drawdown.py` | `core/models/retirement.py` — `RetirementScenario.withdraw_from_holding`; `notebooks/retirement_scenarios.ipynb` |
| 59.5 IRA early-withdrawal gate (skip trad/Roth IRA before age 59.5) | `core/domain/drawdown.py` | `core/models/retirement.py:86` — `withdraw_for_year`; `notebooks/retirement_scenarios.ipynb` |
| RMD-aware withdrawal strategy (net-income-first ordering: earned income → SS → RMDs → taxable → tax-deferred → Roth) | `core/domain/rmd.py` + `core/domain/drawdown.py` | `notebooks/retirement.ipynb` — `calculate_withdrawal_strategy`; the simulation-loop framing is `core/models/retirement.py::simulate` |

### Taxes

| Capability | Target | Source |
|---|---|---|
| FICA (SS 6.2% to wage cap, Medicare 1.45% + 0.9% additional over $250k MFJ) | `core/domain/taxes.py` | `core/models/individual.py:304-355` — `get_fica_wages`/`calculate_net_pay`; `core/models/household.py:214-253` — `compute_taxes` |
| CO state tax (4.25% + 0.45% FAMLI on state wages) | `core/domain/taxes.py` | same as above (`individual.py:347`, `household.py:239`) |
| Wage bases (federal vs FICA vs Medicare vs state: which pre-tax deductions apply to which base — 401k/HSA/employer premium) | `core/domain/taxes.py` | `core/models/individual.py:258-322` — `calculate_pre_tax_deductions`, `calculate_hsa_deductions`, `get_*_wages` |

### Household

| Capability | Target | Source |
|---|---|---|
| Joint-contribution solver (iteratively solves each member's required contribution to joint expenses given asymmetric incomes and the household tax function) | new `core/domain/household.py` | `core/models/household.py:102-185` — `get_joint_contribution_required`, `assign_joint_contributions`; `notebooks/retirement_classes.ipynb`; `notebooks/joint_contribution_debug.ipynb` |
| Household tax allocation (compute MFJ taxes once, allocate back to members/businesses by share) | new `core/domain/household.py` (delegating to `taxes.py`) | `core/models/household.py:202-315` — `compute_taxes`, `allocate_taxes_to_entities` |

### Contributions

| Capability | Target | Source |
|---|---|---|
| Roth IRA MAGI phase-out (`calc_roth_max_contribution`) | `core/domain/assumptions.py` (contribution caps live there) | `notebooks/retirement.ipynb` — "Contributions: Roth IRA" cells |
| 401(k) employer match (matched contributions add to balances but not to take-home reduction) | `core/domain/assumptions.py` | `core/models/individual.py` — `PersonalIncome._employer_match`, `PreTaxContribution(matched=True)` |

### Healthcare

| Capability | Target | Source |
|---|---|---|
| ACA→Medicare premium crossover driven by employer/ACA/Medicare premium inputs per individual (vs. today's fixed HC_* constants) | `core/domain/healthcare.py` | `core/models/healthcare.py` — `HealthCare.get_health_costs`, `get_pre_tax_deductions`. **Known bug pinned in tests:** models version indexes on year where it means age; the domain version already fixed this — port the premium structure, not the indexing. |

### Social Security

| Capability | Target | Source |
|---|---|---|
| `pad_earnings_to_35` (zero-pad an earnings record to 35 years, past- or future-padded by mode) | `core/domain/social_security.py` | `notebooks/retirement.ipynb` — `pad_earnings_to_35` cell |

### Returns / Monte Carlo

| Capability | Target | Source |
|---|---|---|
| Per-holding Monte Carlo (bootstrap each holding from its own ticker history, aggregate) | `core/domain/returns.py` (math) + `core/services/market_data.py` (ticker fetch) | `core/models/portfolio.py` — `Holding.get_historical_returns`, `initialize_forecast_matrix`, `forecast_price`, `Portfolio.bootstrap_portfolio_growth`; `notebooks/portfolio.ipynb` |
| Parametric MC + IQR bands (normal-draw variant, percentile band charting) | `core/domain/returns.py` | `notebooks/portfolio.ipynb`. Note: `returns.py` already has a portfolio-level bootstrap MC — unify rather than duplicate. |

### Business

| Capability | Target | Source |
|---|---|---|
| Business modelling (revenue/expense streams, write-offs, net revenue, income distribution to owners) | new `core/domain/business.py` | `core/models/business.py` — `Business`; `notebooks/retirement.ipynb` — "Business Class" cells |
| `BusinessOwnership` validation (ownership shares sum to 1, distribution rules) | new `core/domain/business.py` | `notebooks/retirement.ipynb` — `BusinessOwnership` cell |

## Supporting machinery (port only if a capability above needs it)

`core/models/core.py` (`Stream`, `FinancialEntity`) and `core/models/transactions.py`
(`Transactions` wrapper: past-year averaging, inflation projection) are Engine A's
plumbing, not capabilities. The domain engine projects from config segments instead
(`core/domain/assumptions.py`). If a ported capability needs transaction-history
projection, extract the specific function; do not port the class hierarchy.

## Deletion map

A `core/models/` file may be deleted when the listed capabilities have all landed:

- `retirement.py` — per-lot basis, 59.5 gate, RMD-aware strategy
- `household.py` — joint-contribution solver, household tax allocation
- `individual.py` — FICA/state/wage bases, employer match (plus whatever
  `household.py` still references)
- `healthcare.py` — ACA→Medicare crossover
- `portfolio.py` — per-holding MC, parametric MC
- `business.py` — business modelling, BusinessOwnership
- `core.py`, `transactions.py`, `__init__.py` — last to go, once nothing above imports them
- `core/utils/functions.py` (the Phase 3 re-export shim) dies with the notebooks'
  migration off it

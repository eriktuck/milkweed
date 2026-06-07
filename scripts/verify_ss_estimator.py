"""Verify the Social-Security-from-income estimator (Phase 6c).

Pure check — no Firestore. Confirms the gate criteria: the PIA matches the
Phase 0 bend-point rules, the W2-vs-1099 covered-earnings difference is correct,
the wage-base cap and short-career averaging behave, and the claim-age factor
flows through `social_security_income`.

Run from the repo root:
    python -m scripts.verify_ss_estimator
"""

from core.services.retirement import (
    SS_BEND_POINTS,
    SS_WAGE_BASE,
    covered_earnings,
    estimate_annual_pia_from_income,
    primary_insurance_amount,
    social_security_income,
)

B1, B2 = SS_BEND_POINTS


def _pia_monthly(aime):
    return 0.9 * min(aime, B1) + 0.32 * max(min(aime, B2) - B1, 0) + 0.15 * max(aime - B2, 0)


def check_bend_points() -> None:
    # Hand-compute PIA at three AIME levels (below b1, between, above b2).
    for aime in (900, 4000, 9000):
        got = primary_insurance_amount(aime)
        assert abs(got - _pia_monthly(aime)) < 1e-6, (aime, got)
    print(f"  bend-point PIA (90/32/15 at {B1:,.0f}/{B2:,.0f}) ✓")


def check_w2_vs_1099() -> None:
    w2 = estimate_annual_pia_from_income(100_000, "W2", 35)
    se = estimate_annual_pia_from_income(100_000, "1099", 35)
    # W2 covers full gross; 1099 covers net × 0.9235 → lower covered earnings → lower PIA.
    assert covered_earnings(100_000, "W2") == 100_000.0
    assert abs(covered_earnings(100_000, "1099") - 92_350.0) < 1e-6
    assert se < w2, (se, w2)
    # Exact W2 value: AIME 8333.33 → PIA 3217.55/mo → ~38,610.6/yr.
    assert abs(w2 - _pia_monthly(100_000 / 12) * 12) < 1e-6
    print(f"  W2 {w2:,.0f}/yr  >  1099 {se:,.0f}/yr  ✓  (self-employment haircut)")


def check_caps_and_career() -> None:
    # Wage base caps covered earnings.
    assert covered_earnings(300_000, "W2") == SS_WAGE_BASE
    capped = estimate_annual_pia_from_income(300_000, "W2", 35)
    assert abs(capped - _pia_monthly(SS_WAGE_BASE / 12) * 12) < 1e-6

    # Short career averages in $0 years → lower benefit than a full career.
    full = estimate_annual_pia_from_income(100_000, "W2", 35)
    short = estimate_annual_pia_from_income(100_000, "W2", 20)
    assert short < full, (short, full)
    assert abs(short - _pia_monthly(100_000 * 20 / 420) * 12) < 1e-6
    print(f"  wage-base cap ✓ ({SS_WAGE_BASE:,.0f})   short career {short:,.0f} < full {full:,.0f} ✓")


def check_claim_factor_flows() -> None:
    pia = estimate_annual_pia_from_income(100_000, "W2", 35)
    ss = social_security_income(pia, claim_age=70, retirement_age=67, death_age=90)
    # Delaying to 70 → 124% of PIA, paid from the claim age (70) onward.
    assert abs(float(ss.loc[70]) - pia * 1.24) < 1e-6, float(ss.loc[70])
    assert float(ss.loc[69]) == 0.0  # nothing before the claim age
    print(f"  claim-age factor flows through ✓  (PIA {pia:,.0f} → {pia * 1.24:,.0f}/yr at 70)")


def main() -> None:
    check_bend_points()
    check_w2_vs_1099()
    check_caps_and_career()
    check_claim_factor_flows()
    print("\nPhase 6c SS-from-income estimator verified ✓")


if __name__ == "__main__":
    main()

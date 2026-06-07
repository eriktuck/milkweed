"""Verify the retirement healthcare glide + LTC spike (Phase 7).

Pure check — no Firestore. Confirms:
  1. The ACA bridge applies before Medicare age and Medicare + OOP applies after
     (indexed on AGE — the fix for core.models.healthcare's age-vs-year bug).
  2. The editable LTC spike adds only across [start, start+years) and the late-life
     total exceeds the go-go baseline (the "smile" upturn).

Run from the repo root:
    python -m scripts.verify_healthcare
"""

from core.services.retirement import (
    HC_ACA_ANNUAL,
    HC_OOP_ANNUAL,
    MEDICARE_AGE,
    MEDICARE_PART_B_MONTHLY_2025,
    healthcare_costs_by_age,
)

PART_B = MEDICARE_PART_B_MONTHLY_2025 * 12.0


def check_aca_to_medicare() -> None:
    # Retire at 62 → ACA bridge for 62..64, Medicare + OOP from 65.
    hc = healthcare_costs_by_age(62, 90)
    assert hc.loc[62] == HC_OOP_ANNUAL + HC_ACA_ANNUAL, hc.loc[62]
    assert hc.loc[64] == HC_OOP_ANNUAL + HC_ACA_ANNUAL
    assert hc.loc[65] == HC_OOP_ANNUAL + PART_B, hc.loc[65]
    assert hc.loc[90] == HC_OOP_ANNUAL + PART_B
    # The bridge is more expensive than Medicare — the cost steps DOWN at 65.
    assert hc.loc[64] > hc.loc[65]
    print(f"  ACA bridge {hc.loc[64]:,.0f} (pre-{MEDICARE_AGE}) → Medicare+OOP "
          f"{hc.loc[65]:,.0f} ✓")


def check_ltc_spike() -> None:
    base = healthcare_costs_by_age(65, 95)
    spiked = healthcare_costs_by_age(65, 95, ltc_annual=70_000, ltc_start_age=85, ltc_years=3)
    # Spike only in [85, 88).
    assert spiked.loc[84] == base.loc[84]
    assert spiked.loc[85] == base.loc[85] + 70_000
    assert spiked.loc[87] == base.loc[87] + 70_000
    assert spiked.loc[88] == base.loc[88]
    # Late-life total now exceeds the go-go baseline → the smile upturn.
    assert spiked.loc[85] > spiked.loc[66]
    print(f"  LTC spike adds 70,000 across 85–87 only ✓  (smile: age 85 "
          f"{spiked.loc[85]:,.0f} > age 66 {spiked.loc[66]:,.0f})")


def main() -> None:
    check_aca_to_medicare()
    check_ltc_spike()
    print("\nPhase 7 healthcare glide + LTC spike verified ✓")


if __name__ == "__main__":
    main()

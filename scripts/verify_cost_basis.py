"""Verify the Vanguard cost-basis parser + taxable-gain math (Phase 6a).

Pure check — no Firestore. Parses the two sample cost-basis exports in data/ and
asserts the per-account gain fractions match the values verified by hand in
.planning/STATE.md (acct …1370 → 37.4%, …8191 → 53.2%), then exercises the pure
gain_fraction_from_cost_basis to confirm taxable-account filtering.

Run from the repo root:
    python -m scripts.verify_cost_basis
"""

from pathlib import Path

from core.services.investments import (
    detect_vanguard_csv_type,
    parse_vanguard_cost_basis_csv,
)
from core.services.retirement import gain_fraction_from_cost_basis

DATA = Path(__file__).resolve().parent.parent / "data"

# Account → expected (gain fraction %, label used in investment_accounts).
EXPECTED = {
    "1370": (37.4, "Taxable"),
    "8191": (53.2, "Taxable"),
}
FILES = {
    "1370": DATA / "costbasisdownload_1370.csv",
    "8191": DATA / "costbasisdownload_8191.csv",
}


def main() -> None:
    cost_basis: dict[str, dict] = {}

    for acct, path in FILES.items():
        content = path.read_text()

        csv_type = detect_vanguard_csv_type(content)
        assert csv_type == "cost_basis", f"{path.name}: detected {csv_type!r}, expected 'cost_basis'"

        records = parse_vanguard_cost_basis_csv(content)
        assert len(records) == 1, f"{path.name}: expected 1 account, got {len(records)}"
        rec = records[0]
        assert rec.account_number == acct, f"{path.name}: account {rec.account_number!r} != {acct!r}"

        frac_pct = rec.gain / rec.market_value * 100
        expected_pct, label = EXPECTED[acct]
        assert abs(frac_pct - expected_pct) < 0.1, (
            f"acct {acct}: gain {frac_pct:.1f}% != expected {expected_pct}%"
        )
        cost_basis[acct] = {
            "market_value": rec.market_value,
            "cost": rec.cost,
            "gain": rec.gain,
        }
        print(f"  acct …{acct}: MV={rec.market_value:,.2f}  cost={rec.cost:,.2f}  "
              f"gain={frac_pct:.1f}%  ✓")

    # Pure fraction over all-taxable accounts → blended fraction.
    accounts = {acct: label for acct, (_, label) in EXPECTED.items()}
    blended = gain_fraction_from_cost_basis(cost_basis, accounts)
    print(f"  blended taxable gain fraction: {blended * 100:.1f}%")

    # Filtering: label both accounts tax-deferred → no taxable cost basis → None.
    none_result = gain_fraction_from_cost_basis(
        cost_basis, {acct: "IRA" for acct in EXPECTED}
    )
    assert none_result is None, f"expected None for all-trad accounts, got {none_result}"
    print("  all-tax-deferred accounts → None (correctly excluded)  ✓")

    print("\nPhase 6a cost-basis parser verified ✓")


if __name__ == "__main__":
    main()

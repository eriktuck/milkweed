from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import json
import re
from core.services.firebase import (
    db,
    find_household_for_user,
    get_user_config
)

_DATE_PAT = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class RetirementConfig(BaseModel):
    """Retirement-page planning assumptions, persisted under UserConfig.retirement.

    Added for the Retirement page (spec-retirement.md, Phase 2). Every field is
    optional with a code-side default in core/services/retirement.py, so a config
    that predates this block behaves as if the user accepted all defaults — no
    Firestore migration is required for read-compatibility (mirrors how
    transaction_account_settings was introduced).

    Field names reuse the existing core/models/individual.py vocabulary
    (`birth_year`, `retirement_age`, `death_age`, `claim_age`) so the v1 light
    service and the future RetirementScenario engine share one set of inputs.

    `birth_year` is the one value that cannot be derived or defaulted (RMD start
    age depends on it — 73 vs 75); the page captures it on first use and it stays
    None until then.
    """
    # Ages / dates -----------------------------------------------------------
    birth_year: int | None = None          # e.g. 1985; drives RMD start age (Phase 6)
    retirement_age: int | None = None      # default 65 (see service DEFAULTS)
    death_age: int | None = None           # planning horizon; default 90
    claim_age: int | None = None           # Social Security claim age; default 67 (FRA)

    # Spending-phase boundaries (ages). go-go = [retirement_age, slow_go_age),
    # slow-go = [slow_go_age, no_go_age), no-go = [no_go_age, death_age].
    slow_go_age: int | None = None         # default 75
    no_go_age: int | None = None           # default 85

    # Economic assumptions ---------------------------------------------------
    # No general inflation field by design: the model is in real (inflation-
    # adjusted) dollars, so `real_return` already nets out inflation and a general
    # inflation knob would double-count. Any future need is purpose-specific
    # (nominal tax-bracket indexing / healthcare excess inflation) and gets its
    # own dedicated field then.
    real_return: float | None = None       # expected real return; default 0.05
    withdrawal_rate: float | None = None    # SWR sanity-check rate; default 0.04

    # Per-csp-key phase multiplier overrides. Shape:
    #   {csp_key: {"slow_go": 0.9, "no_go": 0.7}}  (go-go is always 1.0)
    # Absent keys fall back to PHASE_FACTORS in the service (seeded from research).
    # This is the user-editable crosswalk; the Expenses pane (Phase 5) writes it.
    phase_factors: Dict[str, Dict[str, float]] | None = None

    # Current balances by tax bucket (taxable / trad / roth). Optional override;
    # when None the service derives buckets from the holdings snapshot +
    # investment_accounts (see retirement.balances_by_tax_bucket). Used by the
    # full tax-aware engine in Phase 6.
    account_balances: Dict[str, float] | None = None


class UserConfig(BaseModel):
    accounts: List[str]
    cat_names: Dict[str, str]
    cat_order: List[str]
    csp_from_category: Dict[str, str]
    csp_from_group: Dict[str, str]
    csp_labels: Dict[str, str]
    drop_cats: List[str]
    group_names: Dict[str, str]
    name: str
    uid: str | None = None
    budget: dict | None = None
    csp_plan: dict | None = None  # legacy — superseded by csp_plans
    csp_plans: dict | None = None
    net_worth: dict | None = None
    # Maps last-4 Vanguard account number → account type label.
    # Labels are user-defined, e.g. "IRA", "Roth IRA", "Brokerage", "403b".
    # Drives the retirement vs non-retirement split on the Investments page.
    investment_accounts: Dict[str, str] | None = None
    # Maps last-4 account number → user-chosen nickname, e.g. "Work 403b".
    investment_account_nicknames: Dict[str, str] | None = None
    # Per transaction-account settings, keyed by Monarch account displayName
    # (the same string stored in `accounts` and on each transaction's
    # `account_name`). Each value is {"include": bool, "nickname": str}.
    # `include == False` drops that account's transactions instead of saving
    # them. Absent key → included (backward compatible with pre-control-plane
    # configs that have no settings map).
    transaction_account_settings: Dict[str, dict] | None = None
    # Retirement-page planning assumptions (spec-retirement.md, Phase 2).
    # Absent → service defaults apply; no migration needed (see RetirementConfig).
    retirement: RetirementConfig | None = None
    # ── User Profile (retirement modelling background data) ──
    # All optional; absent → functional defaults supplied by the model/UI layer.
    # Edited on the Profile page (pages/profile.py).
    # NOTE: birth_year/retirement_age/claim_age/death_age also exist on the
    # `retirement` block above (added in parallel) — see overlap to reconcile.
    birth_date: str | None = None              # ISO "YYYY-MM-DD"; birth_year = year part
    coast_age: int | None = None               # default 50; must be < retirement_age
    retirement_age: int | None = None          # default 67
    claim_age: int | None = None               # default 70; Social Security claim, 62–70
    death_age: int | None = None               # default 90; planning horizon
    income_growth_rate: float | None = None    # real annual income growth; default 0.03
    # Gross-income history as forward-filled segments. Each entry is
    # {"date": "YYYY-MM-DD", "amount": <annual gross income>}: the amount is the
    # annual rate in effect from `date` until the next segment. A future-dated
    # segment models a planned raise. See core.utils.functions.segments_to_annual_income.
    income_segments: List[dict] | None = None


class HouseholdConfig(UserConfig):
    members: List[str]


class SessionDataModel(BaseModel):
    """Validated structure for session data."""
    users: Dict[str, UserConfig | HouseholdConfig]


class SessionData:
    """Class to handle session data loading from Firestore."""

    def __init__(self, data: dict | None = None):
        self.data = data or {"users": {}}
        self.logged_in_uid = None
    
    @staticmethod
    def fetch_budgets(ref):
        budgets = {}
        budget_docs = ref.collection("budgets").stream()
        for doc in budget_docs:
            year, month = doc.id.split("-")
            year_dict = budgets.setdefault(year, {})
            year_dict[int(month)] = doc.to_dict()
        return budgets

    @staticmethod
    def fetch_csp_snapshot(ref, key):
        doc = ref.collection("csp_snapshots").document(key).get()
        return doc.to_dict() if doc.exists else {}

    @staticmethod
    def fetch_csp_plans(ref):
        """Stream csp_snapshots; docs with YYYY-MM-DD IDs become csp_plans entries.
        Falls back to the legacy 'plan' doc if no dated entries exist."""
        plans = {}
        for doc in ref.collection("csp_snapshots").stream():
            if _DATE_PAT.match(doc.id):
                plans[doc.id] = doc.to_dict()
        if not plans:
            # backward compat: old flat 'plan' doc → treat as epoch entry
            legacy_doc = ref.collection("csp_snapshots").document("plan").get()
            if legacy_doc.exists:
                plans["1970-01-01"] = legacy_doc.to_dict()
        return plans
    
    def get_user_configs(self) -> dict[str,dict]:
        """Get the user configurations as dictionary"""
        return self.data.get("users", {})
    
    def _load_from_firestore(self, uid):
        """Load household + member data from Firestore."""
        self.logged_in_uid = uid

        # Find the household where the user is a member
        household_id = find_household_for_user(db, uid)
        if household_id:
            # Load household config
            hh_config = get_user_config("households", household_id)
            if not hh_config:
                raise ValueError(f"No configuration found for document {household_id}.")

            # Get household (joint) budgets and settings
            hh_ref = db.collection("households").document(household_id)
            hh_config["uid"] = household_id
            hh_config["budget"] = self.fetch_budgets(hh_ref)
            hh_config["csp_plans"] = self.fetch_csp_plans(hh_ref)
            hh_config["net_worth"] = self.fetch_csp_snapshot(hh_ref, "net_worth")

            self.data["users"][household_id] = hh_config
        
            # Individual budgets and settings for each member
            members = hh_config.get("members", [])
        else:
            members = [uid]
        
        for member_id in members:
            user_config = get_user_config("users", member_id)
            if not user_config:
                print(f"Warning: No configuration found for document {member_id}. Skipping.")
                continue

            user_ref = db.collection("users").document(member_id)
            user_config["uid"] = member_id
            user_config["budget"] = self.fetch_budgets(user_ref)
            user_config["csp_plans"] = self.fetch_csp_plans(user_ref)
            user_config["net_worth"] = self.fetch_csp_snapshot(user_ref, "net_worth")

            self.data["users"][member_id] = user_config

        return self
    
    def get_user_list(self) -> list[str]:
        """Get a list of usernames in the session data."""
        return list(self.data.get("users", {}).keys())
    
    def serialize(self) -> str:
        """Serialize the session data to a JSON string."""
        return json.dumps(self.data, indent=2, ensure_ascii=False)
    
    # ----------------------------
    # Public constructors
    # ----------------------------
    @classmethod
    def from_firestore(cls, uid: str):
        """Convenience constructor to load a new SessionData directly from Firestore."""
        instance = cls()
        instance._load_from_firestore(uid)
        instance._validate()
        return instance

    @classmethod
    def from_json(cls, json_str: str):
        """Recreate a SessionData instance from a JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON provided: {e}")
        instance = cls(data)
        instance._validate()
        return instance
    
    def _validate(self):
        """Validate self.data against the Pydantic schema."""
        SessionDataModel(**self.data)
    
    # ----------------------------
    # File-based I/O
    # ----------------------------
    def save_to_file(self, filepath: str):
        """Save the session data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(data)
        instance._validate()
        return instance
    
    def __repr__(self):
        return f"<SessionData users={len(self.data.get('users', {}))}>"
    

    # # Find the household where the user is a member
    # matching = (
    #     db.collection("households")
    #       .where("members", "array_contains", uid)
    #       .limit(1)
    #       .stream()
    # )
    # household_doc = next(matching, None)
    # if household_doc is None:
    #     raise ValueError("Error: User not assigned to a household")
    
    # household_data = household_doc.to_dict()
    # household_id = household_doc.id
    # members = household_data.get("members", [])

    # # Helper to fetch budgets
    # def fetch_budgets(ref):
    #     budgets = {}
    #     budget_docs = ref.collection("budgets").stream()
    #     for doc in budget_docs:
    #         year, month = doc.id.split("-")
    #         year_dict = budgets.setdefault(year, {})
    #         year_dict[int(month)] = doc.to_dict()
    #     return budgets

    # config = {
    #     "users": {},
    #     "group_names": {},
    #     "cat_names": {},
    #     "account_owner": {}  # <-- Add account_owner mapping!
    # }

    # # Household (joint) budgets and settings
    # household_budgets = fetch_budgets(db.collection("households").document(household_id))
    # config["users"]["joint"] = {
    #     "uid": "joint",
    #     "budget": household_budgets,
    #     "drop_cats": household_data.get("drop_cats", []),
    #     "csp_from_group": household_data.get("csp_from_group", {}),
    #     "csp_from_category": household_data.get("csp_from_category", {}),
    #     "csp_labels": household_data.get("csp_labels", {}),
    #     "cat_order": household_data.get("cat_order", []),
    #     "group_names": household_data.get("group_names", {}),
    #     "cat_names": household_data.get("cat_names", {}),
    #     "accounts": household_data.get("accounts", [])
    # }

    # # Household accounts → owner is 'joint'
    # for acct in household_data.get("accounts", []):
    #     config["account_owner"][acct] = "joint"

    # # Individual budgets and settings
    # for member_id in members:
    #     user_ref = db.collection("users").document(member_id)
    #     user_doc = user_ref.get()
    #     if not user_doc.exists:
    #         continue

    #     user_data = user_doc.to_dict()
    #     username = user_data.get("name", member_id)

    #     # Save group_names and cat_names once
    #     if not config["group_names"]:
    #         config["group_names"] = user_data.get("group_names", {})
    #     if not config["cat_names"]:
    #         config["cat_names"] = user_data.get("cat_names", {})

    #     user_budgets = fetch_budgets(user_ref)

    #     config["users"][username] = {
    #         "uid": member_id,
    #         "budget": user_budgets,
    #         "drop_cats": user_data.get("drop_cats", []),
    #         "csp_from_group": user_data.get("csp_from_group", {}),
    #         "csp_from_category": user_data.get("csp_from_category", {}),
    #         "csp_labels": user_data.get("csp_labels", {}),
    #         "cat_order": user_data.get("cat_order", []),
    #         "group_names": user_data.get("group_names", {}),
    #         "cat_names": user_data.get("cat_names", {}),
    #         "accounts": user_data.get("accounts", [])
    #     }

    #     # User accounts → owner is username
    #     for acct in user_data.get("accounts", []):
    #         config["account_owner"][acct] = username
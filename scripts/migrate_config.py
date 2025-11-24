import firebase_admin
from firebase_admin import credentials, auth, firestore
import json

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-service-account.json")
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Load JSON config file
with open("data/config.json", "r") as f:
    config_data = json.load(f)

# Step 1: Fetch UIDs and create household
user_email_map = {
    "erik": "eriktuck@gmail.com",
    "rachel": "rsurvil@gmail.com"
}
user_ids = []
email_uid_map = {}

for name, email in user_email_map.items():
    try:
        user = auth.get_user_by_email(email)
        user_ids.append(user.uid)
        email_uid_map[email] = user.uid
    except firebase_admin.auth.UserNotFoundError:
        print(f"User with email {email} not found.")
    except Exception as e:
        print(f"Error fetching user for {email}: {e}")

# Create household if both users are found
household_ref = None
if len(user_ids) == len(user_email_map):
    household_data = {
        "name": "The Andersons",
        "members": user_ids
    }
    household_ref = db.collection("households").document()
    household_ref.set(household_data)
    print(f"Created household '{household_data['name']}' with ID: {household_ref.id}")
else:
    raise ValueError("Household not created; one or more users not found.")

# Shared config settings
group_names = config_data.get("group_names", {})
cat_names = config_data.get("cat_names", {})
account_owner = config_data.get("account_owner", {})

# Step 2: Migrate user budgets and config
users_data = config_data.get("users", {})
for user_key, user_data in users_data.items():
    if user_key == "joint":
        base_ref = household_ref
        accounts = [acct for acct, owner in account_owner.items() if owner == user_key]
        base_ref.set({"accounts": accounts}, merge=True)
    
    else:
        uid = email_uid_map.get(user_email_map.get(user_key))
        if not uid:
            print(f"Skipping user '{user_key}' - no matching UID found.")
            continue
        base_ref = db.collection("users").document(uid)

        # Add accounts from account_owner
        accounts = [acct for acct, owner in account_owner.items() if owner == user_key]
        base_ref.set({"accounts": accounts}, merge=True)

        # Save name
        base_ref.set({"name": user_key}, merge=True)

    # Store budgets
    budget = user_data.get("budget", {})
    for year, months in budget.items():
        for month, categories in months.items():
            doc_id = f"{year}-{str(month).zfill(2)}"
            base_ref.collection("budgets").document(doc_id).set(categories)

    # Store config values
    config_fields = ["drop_cats", "csp_from_group", "csp_from_category", "csp_labels", "cat_order"]
    config_payload = {k: v for k, v in user_data.items() if k in config_fields}
    config_payload["group_names"] = group_names
    config_payload["cat_names"] = cat_names
    base_ref.set(config_payload, merge=True)

print("Migration complete.")

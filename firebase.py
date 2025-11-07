import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Prevent multiple initializations
if not firebase_admin._apps:
    firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")

    if firebase_credentials_json:
        # Load credentials from the environment variable (Cloud Run)
        try:
            cred = credentials.Certificate(json.loads(firebase_credentials_json))
            firebase_admin.initialize_app(cred)
            print("Firebase app initialized from environment variable.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding Firebase credentials from environment variable.")
    else:
        # Load credentials from the local file (local development)
        try:
            cred = credentials.Certificate(
                os.path.join("secrets", "firebase-service-account")
            )
            firebase_admin.initialize_app(cred)
            print("Firebase app initialized from local file.")
        except FileNotFoundError:
            raise ValueError(
                "Firebase service account file not found at secrets/firebase-service-account"
            )
        except Exception as e:
            raise ValueError(f"Error loading Firebase credentials from file: {e}")

db = firestore.client()
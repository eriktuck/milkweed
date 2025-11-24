from core.models.session import SessionData
from pathlib import Path

UID = "Ij893k3NoQSUc5aFmIwj4xKNKzP2"  # erik

def main():
    session_cache = SessionData.from_firestore(UID)

    # Save to output file
    fname = 'session_cache.json'
    fpath = Path("data") / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    session_cache.save_to_file(fpath)

if __name__ == "__main__":
    main()
# modules/csv_utils.py
import csv
import os
from datetime import datetime

USERS_CSV = "users.csv"
VOTES_CSV = "votes.csv"
FRAUD_CSV = "fraud_logs.csv"

def ensure_users_csv():
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name","aadhar","password_hash","folder","embedding_pkl","voted"])

def add_user(name, aadhar, password_hash, folder, embedding_pkl=""):
    ensure_users_csv()
    with open(USERS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, aadhar, password_hash, folder, embedding_pkl, "No"])

def get_user_by_aadhar(aadhar):
    ensure_users_csv()
    with open(USERS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["aadhar"] == str(aadhar):
                return r
    return None

def update_user_embedding(aadhar, embedding_pkl):
    rows = []
    updated = False
    ensure_users_csv()
    with open(USERS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["aadhar"] == str(aadhar):
                r["embedding_pkl"] = embedding_pkl
                updated = True
            rows.append(r)
    if updated:
        with open(USERS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name","aadhar","password_hash","folder","embedding_pkl","voted"])
            writer.writeheader()
            writer.writerows(rows)
    return updated

def set_user_voted(aadhar):
    rows = []
    ensure_users_csv()
    with open(USERS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["aadhar"] == str(aadhar):
                r["voted"] = "Yes"
            rows.append(r)
    with open(USERS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","aadhar","password_hash","folder","embedding_pkl","voted"])
        writer.writeheader()
        writer.writerows(rows)

# ------------------------------
# Votes CSV helpers
# ------------------------------
def ensure_votes_csv():
    if not os.path.exists(VOTES_CSV):
        with open(VOTES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["aadhar","name","party","timestamp"])

def append_vote(aadhar, name, party):
    ensure_votes_csv()
    from datetime import datetime
    with open(VOTES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([aadhar, name, party, datetime.now().isoformat()])

def read_all_votes():
    ensure_votes_csv()
    votes = []
    with open(VOTES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            votes.append(r)
    return votes

def votes_count_by_party():
    counts = {}
    import csv

    with open("votes.csv", newline='') as f:
        reader = csv.DictReader(f)

        # Use the actual key in your CSV
        party_key = 'Party A'  # <-- change to match your CSV key

        for row in reader:
            if party_key in row and row[party_key].strip() != "":
                counts[row[party_key]] = counts.get(row[party_key], 0) + 1
            else:
                print("Skipping invalid row:", row)

    return counts


# ------------------------------
# Fraud logging
# ------------------------------
def ensure_fraud_csv():
    if not os.path.exists(FRAUD_CSV):
        with open(FRAUD_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["aadhar","name","activity","similarity","extra","timestamp"])

def log_fraud(aadhar, name, activity, similarity=None, extra=""):
    ensure_fraud_csv()
    ts = datetime.now().isoformat()
    with open(FRAUD_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([aadhar, name, activity, "" if similarity is None else f"{similarity:.6f}", extra, ts])

def read_fraud_logs(limit=None):
    ensure_fraud_csv()
    logs = []
    with open(FRAUD_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            logs.append(r)
    if limit:
        return logs[-limit:]
    return logs

# ------------------------------
# User list helper
# ------------------------------
def read_all_users():
    ensure_users_csv()
    users = []
    with open(USERS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            users.append(r)
    return users

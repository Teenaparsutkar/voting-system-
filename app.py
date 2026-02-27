# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import hashlib

# --- MODULE IMPORTS ---
from modules.collect_images import capture_images
from modules.train_embeddings import generate_embeddings_for_user
from modules.recognize_face import verify_face_live
from modules.csv_utils import (
    ensure_users_csv,
    add_user,
    get_user_by_aadhar,
    update_user_embedding,
    set_user_voted,
    append_vote,
    read_all_users,
    votes_count_by_party,
    log_fraud,
    read_fraud_logs,
    read_all_votes
)

# --- FLASK SETUP ---
app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"

# --- ENSURE FOLDERS EXIST ---
os.makedirs("dataset", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
ensure_users_csv()

# --- HOME ---
@app.route("/")
def home():
    return render_template("home.html")

# --- REGISTRATION ---
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name").strip()
        aadhar = request.form.get("aadhar").strip()
        password = request.form.get("password")
        if not (name and aadhar and password):
            flash("Please fill all fields.")
            return redirect(url_for("register"))

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        folder = os.path.join("dataset", f"{name}_{aadhar}")
        add_user(name, aadhar, password_hash, folder)
        flash("User added. Next: capture images.")
        return redirect(url_for("capture", name=name, aadhar=aadhar))

    return render_template("register.html")

# --- CAPTURE FACE IMAGES ---
@app.route("/capture/<name>/<aadhar>", methods=["GET","POST"])
def capture(name, aadhar):
    if request.method == "POST":
        # Capture face images (MTCNN now integrated)
        user_folder = capture_images(name, aadhar, num_images=100)
        flash(f"Images saved to {user_folder}. Now generate embeddings.")
        return redirect(url_for("train_page", name=name, aadhar=aadhar))

    return render_template("capture.html", name=name, aadhar=aadhar)

# --- GENERATE EMBEDDINGS ---
@app.route("/train/<name>/<aadhar>", methods=["GET","POST"])
def train_page(name, aadhar):
    if request.method == "POST":
        # Generate embeddings and save .pkl
        pkl_path = generate_embeddings_for_user(name, aadhar)
        update_user_embedding(aadhar, pkl_path)
        flash("Embeddings generated and saved.")
        return redirect(url_for("home"))

    return render_template("train.html", name=name, aadhar=aadhar)

# --- LOGIN ---
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        aadhar = request.form.get("aadhar").strip()
        password = request.form.get("password")

        user = get_user_by_aadhar(aadhar)
        if not user:
            flash("User not found")
            return redirect(url_for("login"))

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != user["password_hash"]:
            flash("Invalid password")
            return redirect(url_for("login"))

        # store aadhar in session while verifying
        session["pending_aadhar"] = aadhar
        return redirect(url_for("verify", name=user["name"], aadhar=aadhar))

    return render_template("login.html")

# --- FACE VERIFICATION ---
@app.route("/verify/<name>/<aadhar>", methods=["GET","POST"])
def verify(name, aadhar):
    if request.method == "POST":
        res = verify_face_live(name, aadhar, threshold=0.6)
        if res["matched"]:
            # if matched but suspicious reason -> log fraud
            if res["reason"] in ("matched_but_low_texture", "low_texture_spoof"):
                log_fraud(aadhar, name, "Low-texture spoof suspicion on verification", similarity=res["similarity"], extra=f"lap_var={res.get('spoof_score')}")
            # proceed to voting
            return redirect(url_for("vote", name=name, aadhar=aadhar))
        else:
            # record failed attempt as potential fraud
            log_fraud(aadhar, name, "Failed verification attempt", similarity=res.get("similarity"), extra=res.get("reason",""))
            flash("Face not recognized. Try again.")
            return redirect(url_for("login"))

    return render_template("verify.html", name=name, aadhar=aadhar)

# --- VOTING PAGE ---
@app.route("/vote/<name>/<aadhar>", methods=["GET","POST"])
def vote(name, aadhar):
    # quick duplicate-vote check
    user = get_user_by_aadhar(aadhar)
    if not user:
        flash("User not found.")
        return redirect(url_for("home"))

    if user.get("voted", "No") == "Yes":
        flash("You have already voted. Duplicate voting is not allowed.")
        # log duplicate attempt
        log_fraud(aadhar, user["name"], "Duplicate vote attempt", similarity=None, extra="tried to access /vote")
        return render_template("error.html", message="Duplicate vote attempt detected.")

    parties = ["Party A", "Party B", "Party C"]
    if request.method == "POST":
        selected = request.form.get("party")
        # Save vote to CSV using helper
        append_vote(aadhar, name, selected)
        # Mark user as voted
        set_user_voted(aadhar)
        return render_template("success.html", party=selected)

    return render_template("vote.html", parties=parties, name=name)

# ------------------------------
# ADMIN 
# ------------------------------
ADMIN_USERNAME = "admin"
# change this password before deploy
ADMIN_PASSWORD_HASH = hashlib.sha256("adminpass".encode()).hexdigest()

@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if u == ADMIN_USERNAME and hashlib.sha256(p.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials")
            return redirect(url_for("admin_login"))
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    flash("Logged out")
    return redirect(url_for("home"))

def require_admin():
    if not session.get("admin"):
        flash("Admin access required")
        return False
    return True

@app.route("/admin/dashboard")
def admin_dashboard():
    if not require_admin():
        return redirect(url_for("admin_login"))

    vote_counts = votes_count_by_party()
    users = read_all_users()
    frauds = read_fraud_logs()
    total_votes = len(read_all_votes())
    return render_template("admin_dashboard.html",
                           vote_counts=vote_counts,
                           users=users,
                           frauds=frauds,
                           total_votes=total_votes)

@app.route("/admin/frauds")
def admin_frauds():
    if not require_admin():
        return redirect(url_for("admin_login"))
    frauds = read_fraud_logs()
    return render_template("admin_frauds.html", frauds=frauds)

@app.route("/admin/users")
def admin_users():
    if not require_admin():
        return redirect(url_for("admin_login"))
    users = read_all_users()
    return render_template("admin_users.html", users=users)

# --- RUN FLASK APP ---
if __name__ == "__main__":
    app.run(debug=True)

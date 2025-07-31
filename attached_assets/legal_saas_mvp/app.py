"""
Simple AI‑powered practice management micro‑SaaS MVP.

This Flask application implements a minimal law firm practice
management tool tailored for solo and small firms.  It illustrates
how a MicroSaaS could be built quickly using Python and open‑source
libraries.  Key features include:

* User authentication (sign up/log in) using hashed passwords.
* Management of clients, cases, and tasks associated with cases.
* A basic AI summarisation endpoint powered by the Hugging Face
  ``transformers`` library to condense long case descriptions into
  brief abstracts.  The summariser runs locally and does not
  require external API keys.  See the ``summarise_text`` function
  below for details.
* Simple bootstrap‑based templates for a responsive UI.

Note: This code is intended for demonstration purposes only.  It is
**not** a substitute for professional legal software and does not
provide legal advice.  Use at your own risk.  Real deployments
should include HTTPS, robust authentication, logging, input
validation, and compliance with relevant data protection laws.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from flask import (Flask, render_template, redirect, url_for, request,
                   session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# Try importing summarisation pipeline.  If transformers isn't
# available the summariser will fallback to returning the first
# sentence of the input.
try:
    from transformers import pipeline  # type: ignore
    summariser = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    summariser = None


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def create_app() -> Flask:
    """Factory to create and configure the Flask app."""
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///" + os.path.join(BASE_DIR, "mvp.db")
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # Secret key for session handling; in production override via
    # environment variable.
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key")

    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app


# Initialise SQLAlchemy outside of factory so models can refer to it
db = SQLAlchemy()


# Database models
class User(db.Model):  # type: ignore[misc]
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    cases = db.relationship("Case", backref="owner", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Case(db.Model):  # type: ignore[misc]
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    client_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tasks = db.relationship("Task", backref="case", lazy=True)

    def summary(self) -> str:
        """Generate a short summary of the case description using the
        summarisation pipeline.  Falls back to first 100 chars if the
        model is unavailable.  Summaries are limited to 120 words.
        """
        if summariser:
            try:
                result = summariser(
                    self.description,
                    max_length=120,
                    min_length=30,
                    do_sample=False,
                )
                return result[0]["summary_text"]
            except Exception:
                pass
        # Fallback: return first sentence or 100 chars
        first_sentence = self.description.split(".")[0]
        return (first_sentence[:100] + "...") if len(first_sentence) > 100 else first_sentence


class Task(db.Model):  # type: ignore[misc]
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(300), nullable=False)
    due_date = db.Column(db.Date, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)


# Flask views
app = create_app()


def login_required(fn):  # type: ignore[call-arg]
    """Simple decorator to require authentication for a route."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if not email or not password:
            flash("Email and password required", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for("register"))
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            flash("Logged in successfully", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = User.query.get(session["user_id"])
    return render_template("dashboard.html", cases=user.cases)


@app.route("/case/new", methods=["GET", "POST"])
@login_required
def new_case():
    if request.method == "POST":
        title = request.form.get("title")
        client_name = request.form.get("client_name")
        description = request.form.get("description")
        if not title or not client_name or not description:
            flash("All fields are required", "danger")
            return redirect(url_for("new_case"))
        case = Case(
            title=title,
            client_name=client_name,
            description=description,
            owner_id=session["user_id"],
        )
        db.session.add(case)
        db.session.commit()
        flash("Case created", "success")
        return redirect(url_for("dashboard"))
    return render_template("new_case.html")


@app.route("/case/<int:case_id>")
@login_required
def view_case(case_id: int):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != session.get("user_id"):
        flash("Access denied", "danger")
        return redirect(url_for("dashboard"))
    return render_template("view_case.html", case=case)


@app.route("/case/<int:case_id>/task/new", methods=["POST"])
@login_required
def new_task(case_id: int):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != session.get("user_id"):
        flash("Access denied", "danger")
        return redirect(url_for("dashboard"))
    description = request.form.get("description")
    due_date_str = request.form.get("due_date")
    if not description or not due_date_str:
        flash("Description and due date required", "danger")
        return redirect(url_for("view_case", case_id=case.id))
    try:
        due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
    except ValueError:
        flash("Invalid date format", "danger")
        return redirect(url_for("view_case", case_id=case.id))
    task = Task(description=description, due_date=due_date, case_id=case.id)
    db.session.add(task)
    db.session.commit()
    flash("Task added", "success")
    return redirect(url_for("view_case", case_id=case.id))


@app.route("/task/<int:task_id>/complete")
@login_required
def complete_task(task_id: int):
    task = Task.query.get_or_404(task_id)
    if task.case.owner_id != session.get("user_id"):
        flash("Access denied", "danger")
        return redirect(url_for("dashboard"))
    task.completed = not task.completed
    db.session.commit()
    return redirect(url_for("view_case", case_id=task.case.id))


@app.route("/summarise/<int:case_id>")
@login_required
def summarise(case_id: int):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != session.get("user_id"):
        flash("Access denied", "danger")
        return redirect(url_for("dashboard"))
    summary = case.summary()
    return render_template("summary.html", case=case, summary=summary)


if __name__ == "__main__":
    # Run development server
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
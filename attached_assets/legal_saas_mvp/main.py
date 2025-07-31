"""
FastAPI‑based micro‑SaaS MVP for small law firms.

This application demonstrates a minimal practice management system built
with built‑in Python libraries and FastAPI, requiring no external
packages beyond what's preinstalled in the environment.  It offers
basic user authentication, case and task management, and a simple
summarisation endpoint.  The summarisation is a naive heuristic
(returning the first two sentences) because offline constraints
prevent us from leveraging modern AI models.  In a production
environment, you would replace this with a proper natural language
model or API call.

Important: This code is for demonstration only and is not a
production‑ready solution.  It lacks encryption, CSRF protection,
and comprehensive security hardening.  Do not use this as‑is in
production, especially for handling sensitive legal data.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
import hashlib
import hmac
from datetime import datetime, date
from typing import Dict, Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


DATABASE_PATH = os.path.join(os.path.dirname(__file__), "mvp.sqlite3")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# In‑memory session store.  Maps session token -> user_id.
sessions: Dict[str, int] = {}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        );
        """
    )
    # Cases table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            client_name TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TEXT NOT NULL,
            owner_id INTEGER NOT NULL,
            FOREIGN KEY(owner_id) REFERENCES users(id)
        );
        """
    )
    # Tasks table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            due_date TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            case_id INTEGER NOT NULL,
            FOREIGN KEY(case_id) REFERENCES cases(id)
        );
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a given salt using SHA256."""
    return hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()


def create_user(email: str, password: str) -> Optional[int]:
    salt = uuid.uuid4().hex
    pw_hash = hash_password(password, salt)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password_hash, salt) VALUES (?, ?, ?)",
            (email, pw_hash, salt),
        )
        conn.commit()
        user_id = cur.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[int]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, salt FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row:
        pw_hash = hash_password(password, row["salt"])
        if hmac.compare_digest(pw_hash, row["password_hash"]):
            return row["id"]
    return None


def get_user_cases(user_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, client_name, description, created_at FROM cases WHERE owner_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    cases = cur.fetchall()
    conn.close()
    return cases


def get_case(case_id: int) -> Optional[sqlite3.Row]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, client_name, description, created_at, owner_id FROM cases WHERE id = ?",
        (case_id,),
    )
    case = cur.fetchone()
    conn.close()
    return case


def get_case_tasks(case_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, description, due_date, completed FROM tasks WHERE case_id = ? ORDER BY due_date",
        (case_id,),
    )
    tasks = cur.fetchall()
    conn.close()
    return tasks


def add_case(owner_id: int, title: str, client_name: str, description: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO cases (title, client_name, description, created_at, owner_id) VALUES (?, ?, ?, ?, ?)",
        (title, client_name, description, datetime.utcnow().isoformat(), owner_id),
    )
    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def add_task(case_id: int, description: str, due_date: date) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tasks (description, due_date, case_id) VALUES (?, ?, ?)",
        (description, due_date.isoformat(), case_id),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def toggle_task_completion(task_id: int) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT completed FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    new_status = 0 if row["completed"] else 1
    cur.execute(
        "UPDATE tasks SET completed = ? WHERE id = ?",
        (new_status, task_id),
    )
    conn.commit()
    conn.close()


def summarise_text(text: str, max_sentences: int = 2) -> str:
    """Return a naive summary: the first `max_sentences` sentences from the text."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("..." if len(sentences) > max_sentences else "")


@app.on_event("startup")
def startup_event() -> None:
    init_db()


def get_current_user(request: Request) -> Optional[int]:
    session_token = request.cookies.get("session_token")
    if session_token and session_token in sessions:
        return sessions[session_token]
    return None


def require_login(request: Request) -> int:
    user_id = get_current_user(request)
    if user_id is None:
        raise HTTPException(status_code=303, detail="Redirect to login", headers={"Location": "/login"})
    return user_id


@app.get("/", response_class=HTMLResponse, name="index")
def landing(request: Request):
    user_id = get_current_user(request)
    if user_id:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register", response_class=HTMLResponse)
def register_post(request: Request, email: str = Form(...), password: str = Form(...)):
    if not email or not password:
        return templates.TemplateResponse(
            "register.html", {"request": request, "error": "Email and password required"}
        )
    user_id = create_user(email, password)
    if user_id is None:
        return templates.TemplateResponse(
            "register.html", {"request": request, "error": "Email already registered"}
        )
    response = RedirectResponse(url="/login", status_code=303)
    return response


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    user_id = authenticate_user(email, password)
    if user_id is None:
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Invalid credentials"}
        )
    # Create session token
    token = uuid.uuid4().hex
    sessions[token] = user_id
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("session_token", token, httponly=True, max_age=60 * 60 * 24)
    return response


@app.get("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token and token in sessions:
        del sessions[token]
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user_id = require_login(request)
    cases = get_user_cases(user_id)
    # compute naive summary for each case in template if needed
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "cases": cases, "summarise": summarise_text}
    )


@app.get("/case/new", response_class=HTMLResponse)
def new_case_get(request: Request):
    require_login(request)
    return templates.TemplateResponse("new_case.html", {"request": request})


@app.post("/case/new")
def new_case_post(
    request: Request,
    title: str = Form(...),
    client_name: str = Form(...),
    description: str = Form(...),
):
    user_id = require_login(request)
    add_case(user_id, title, client_name, description)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/case/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: int):
    user_id = require_login(request)
    case = get_case(case_id)
    if case is None or case["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Case not found")
    tasks = get_case_tasks(case_id)
    return templates.TemplateResponse(
        "view_case.html",
        {
            "request": request,
            "case": case,
            "tasks": tasks,
        },
    )


@app.post("/case/{case_id}/task/new")
def add_task_route(request: Request, case_id: int, description: str = Form(...), due_date: str = Form(...)):
    user_id = require_login(request)
    case = get_case(case_id)
    if case is None or case["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Case not found")
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    add_task(case_id, description, due)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


@app.get("/task/{task_id}/complete")
def toggle_task_route(request: Request, task_id: int):
    user_id = require_login(request)
    # Find the task's case and check ownership
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT case_id FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")
    case_id = row["case_id"]
    case = get_case(case_id)
    if case is None or case["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Case not found")
    toggle_task_completion(task_id)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


@app.get("/summarise/{case_id}", response_class=HTMLResponse)
def summarise_case(request: Request, case_id: int):
    user_id = require_login(request)
    case = get_case(case_id)
    if case is None or case["owner_id"] != user_id:
        raise HTTPException(status_code=404, detail="Case not found")
    summary = summarise_text(case["description"])
    return templates.TemplateResponse(
        "summary.html", {"request": request, "case": case, "summary": summary}
    )


# Mount static directory if needed (currently unused)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

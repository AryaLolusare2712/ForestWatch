"""Local account authentication for the optional React dashboard."""
from __future__ import annotations
import hashlib
import hmac
import secrets
from .database import SessionLocal, User, initialise_database

def _hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000).hex()
    return f"{salt}${digest}"

def _verify(password: str, stored: str) -> bool:
    salt, expected = stored.split("$", 1)
    return hmac.compare_digest(_hash(password, salt), stored)

def register(email: str, password: str) -> dict:
    email = email.strip().lower()
    if "@" not in email or len(password) < 8:
        raise ValueError("Use a valid email address and a password with at least 8 characters.")
    initialise_database(); session = SessionLocal()
    try:
        if session.query(User).filter_by(email=email).first():
            raise ValueError("An account already exists for this email. Please sign in.")
        user = User(email=email, password_hash=_hash(password), session_token=secrets.token_urlsafe(32))
        session.add(user); session.commit()
        return {"email": user.email, "token": user.session_token}
    finally: session.close()

def login(email: str, password: str) -> dict:
    initialise_database(); session = SessionLocal()
    try:
        user = session.query(User).filter_by(email=email.strip().lower()).first()
        if not user or not _verify(password, user.password_hash):
            raise ValueError("Email or password is incorrect.")
        user.session_token = secrets.token_urlsafe(32); session.commit()
        return {"email": user.email, "token": user.session_token}
    finally: session.close()

def user_from_token(token: str | None):
    if not token: return None
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(session_token=token).first()
        return {"id": user.id, "email": user.email} if user else None
    finally: session.close()

"""SMTP delivery of health alerts to the signed-in dashboard user."""
from __future__ import annotations
from datetime import datetime, timezone
from email.message import EmailMessage
import smtplib
from .config import settings
from .database import Alert, AlertDelivery, SessionLocal

def deliver_new_alerts(user: dict) -> dict:
    if not all([settings.smtp_host, settings.smtp_username, settings.smtp_password, settings.smtp_from]):
        return {"sent": 0, "configured": False, "message": "SMTP email is not configured."}
    session = SessionLocal()
    try:
        alerts = session.query(Alert).filter(Alert.status.in_(["OPEN", "IN PROGRESS"])).all()
        pending = [a for a in alerts if not session.query(AlertDelivery).filter_by(alert_id=a.alert_id, user_id=user["id"]).first()]
        if not pending: return {"sent": 0, "configured": True, "message": "No new alerts to email."}
        message = EmailMessage(); message["Subject"] = f"ForestWatch: {len(pending)} vegetation health alert(s)"; message["From"] = settings.smtp_from; message["To"] = user["email"]
        message.set_content("ForestWatch vegetation health alert(s):\n\n" + "\n".join(f"{a.date}: {a.reason}\nAction: {a.recommended_action}" for a in pending))
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as client:
                client.starttls(); client.login(settings.smtp_username, settings.smtp_password); client.send_message(message)
        except (smtplib.SMTPException, OSError):
            # Do not expose email credentials or raw provider errors in the UI.
            return {"sent": 0, "configured": True,
                    "message": "Email was not delivered. For Gmail, use a Google App Password, enable two-step verification, then restart the backend."}
        for alert in pending: session.add(AlertDelivery(alert_id=alert.alert_id, user_id=user["id"], delivered_at=datetime.now(timezone.utc).isoformat()))
        session.commit(); return {"sent": len(pending), "configured": True, "message": f"Sent {len(pending)} alert email(s) to {user['email']}."}
    finally: session.close()

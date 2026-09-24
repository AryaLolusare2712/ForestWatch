"""Health-score alert creation and retrieval."""
from __future__ import annotations
import pandas as pd
from .database import Alert, SessionLocal, initialise_database

HEALTH_ALERT_THRESHOLD = 35.0

def check_health_alerts(scores: pd.DataFrame) -> pd.DataFrame:
    """Store one alert per observation whose computed health score is below 35."""
    initialise_database()
    below = scores.dropna(subset=["forest_health_score", "date"])
    below = below[below.forest_health_score < HEALTH_ALERT_THRESHOLD].drop_duplicates("date", keep="last")
    session = SessionLocal()
    try:
        # Existing SQLite alerts may have been created under an older threshold.
        # Close any of those that no longer satisfy the current rule.
        existing = session.query(Alert).filter(Alert.alert_id.like("health-score-%")).all()
        for item in existing:
            try:
                stored_score = float(item.reason.split(" is ", 1)[1].split(",", 1)[0])
                if stored_score >= HEALTH_ALERT_THRESHOLD:
                    item.status = "RESOLVED"
            except (IndexError, ValueError):
                pass
        for row in below.itertuples():
            alert_id = f"health-score-{row.date}"
            alert = session.query(Alert).filter_by(alert_id=alert_id).first()
            severity = "CRITICAL" if row.forest_health_score < 20 else "WARNING"
            reason = f"Vegetation health score is {row.forest_health_score:.2f}, below the alert threshold of {HEALTH_ALERT_THRESHOLD:.0f}."
            if alert is None:
                session.add(Alert(alert_id=alert_id, date=str(row.date), severity=severity, reason=reason,
                                  ndvi_change=float(row.ndvi_change) if pd.notna(row.ndvi_change) else None,
                                  recommended_action="Review the satellite scene and arrange ground verification if the signal persists.", status="OPEN"))
            else:
                alert.severity, alert.reason = severity, reason
        session.commit()
        rows = session.query(Alert).filter(Alert.alert_id.like("health-score-%"), Alert.status != "RESOLVED").order_by(Alert.date.desc()).all()
        return pd.DataFrame([{"Date": item.date, "Health score": round(float(item.reason.split(" is ")[1].split(",")[0]), 2),
                              "Severity": item.severity, "Reason": item.reason, "Recommended action": item.recommended_action,
                              "Status": item.status} for item in rows])
    finally:
        session.close()

def update_alert_status(alert_date: str, status: str) -> str:
    """Set the workflow status of a current vegetation-health alert."""
    allowed = {"OPEN", "IN PROGRESS", "CLOSED"}
    if status not in allowed:
        return "Choose Open, In Progress, or Closed."
    session = SessionLocal()
    try:
        item = session.query(Alert).filter_by(alert_id=f"health-score-{alert_date}").first()
        if item is None:
            return "No health alert exists for the selected date. Refresh alerts first."
        item.status = status
        session.commit()
        return f"Alert for {alert_date} changed to {status.title()}."
    finally:
        session.close()

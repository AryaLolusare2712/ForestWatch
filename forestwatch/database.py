from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

settings.ensure_output_dirs()
engine=create_engine(f"sqlite:///{settings.output_path / 'forestwatch.db'}", connect_args={"check_same_thread": False})
SessionLocal=sessionmaker(bind=engine)
Base=declarative_base()

class Alert(Base):
    __tablename__="alerts"
    id=Column(Integer,primary_key=True); alert_id=Column(String,unique=True); date=Column(String); severity=Column(String); reason=Column(Text); ndvi_change=Column(Float); recommended_action=Column(Text); status=Column(String,default="OPEN")

class User(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True); email=Column(String,unique=True,nullable=False); password_hash=Column(String,nullable=False); session_token=Column(String,unique=True,nullable=True)

class AlertDelivery(Base):
    __tablename__="alert_deliveries"
    id=Column(Integer,primary_key=True); alert_id=Column(String,nullable=False); user_id=Column(Integer,nullable=False); delivered_at=Column(String,nullable=False)

def initialise_database(): Base.metadata.create_all(engine)

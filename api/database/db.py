from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from datetime import datetime
import os


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    image_name = Column(String(255))
    predicted_class = Column(String(50))
    confidence = Column(Float)
    top5 = Column(Text)
    explanation = Column(Text)
    warning = Column(Text, nullable=True)
    report = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./simclr.db')
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
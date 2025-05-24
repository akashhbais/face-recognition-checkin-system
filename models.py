import uuid
from datetime import datetime
from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "postgresql://postgres:root@localhost/face_db"

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    
    employee_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    image_path = Column(String, unique=True, nullable=False)
    arcface_embedding = Column(JSON, nullable=False)
    vit_embedding = Column(JSON, nullable=False)

class CheckIn(Base):
    __tablename__ = "checkins"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("employees.employee_id"), nullable=False)
    checkin_time = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee")

# Database connection setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

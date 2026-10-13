from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    department = Column(String, nullable=True, index=True)
    role = Column(String, nullable=True, index=True)  # role that can view
    tags = Column(JSON, nullable=True)  # list of strings
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # Store embedding as binary (optional) or JSON list
    embedding = Column(JSON, nullable=True)

from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List

# Database setup
DATABASE_URL = "sqlite:///./chatbot.db"  # Use PostgreSQL or MySQL in production
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)

# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# FastAPI setup
app = FastAPI()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models for Request Data
class UserCreate(BaseModel):
    user_id: str

class ConversationCreate(BaseModel):
    user_id: str
    message: str
    response: str

# Create a User
@app.post("/users/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(user_id=user.user_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Get all Users
@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

# Store a Conversation
@app.post("/conversations/")
def create_conversation(convo: ConversationCreate, db: Session = Depends(get_db)):
    conversation = Conversation(user_id=convo.user_id, message=convo.message, response=convo.response)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return {"message": "Conversation added", "conversation": conversation}

# Get All Conversations for a User
@app.get("/conversations/{user_id}")
def get_conversations_by_user(user_id: str, db: Session = Depends(get_db)):
    conversations = db.query(Conversation).filter(Conversation.user_id == user_id).all()
    return [{"id": conv.id, "user_id": conv.user_id, "message": conv.message, "response": conv.response} for conv in conversations]

# Run using: uvicorn database_service:app --host 0.0.0.0 --port 5006

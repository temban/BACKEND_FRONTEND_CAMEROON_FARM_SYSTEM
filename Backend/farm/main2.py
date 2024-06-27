import databases
import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import sessionmaker, Session

# FastAPI app instance
app = FastAPI()

# SQLAlchemy database URL
DATABASE_URL = "postgresql://postgres:allpha01@localhost/postgres"

# Create a database instance
database = databases.Database(DATABASE_URL)

# Create a SQLAlchemy engine
engine = sqlalchemy.create_engine(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define a base class for ORM models
Base = declarative_base()

# Define a User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    password = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create a sessionmaker instance
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create a new database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    role: str

@app.post("/signup")
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password before storing it in the database
    hashed_password = pwd_context.hash(user_data.password)

    # Create a new user instance
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        password=hashed_password,
        role=user_data.role
    )

    # Add the new user to the session and commit the transaction
    db.add(new_user)
    db.commit()

    return {"user_id": new_user.id}
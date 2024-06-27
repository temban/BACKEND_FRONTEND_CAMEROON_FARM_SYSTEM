from fastapi import status, Form
import cv2
from tensorflow.keras.models import load_model
import warnings
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import random
import string
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
from io import StringIO
from datetime import datetime
import databases
import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Query
from pydantic import BaseModel
from passlib.context import CryptContext
import requests
from datetime import datetime, timedelta
import pusher
from sqlalchemy.sql import func


def generate_random_token():
    portion_length = 120
    portions = []
    for _ in range(3):
        portion = ''.join(random.choices(string.ascii_letters + string.digits, k=portion_length))
        portions.append(portion)
    return '.'.join(portions)

def publish_message(message):
    pusher_client = pusher.Pusher(
        app_id='1663724',
        key='a7d67608f236b9565325',
        secret='b83f97c2ef83e7d7dbbf',
        cluster='eu',
        ssl=True
    )

    pusher_client.trigger('my-channel', 'my-event', {'message': message})

# Generate random token
random_token = generate_random_token()

warnings.filterwarnings("ignore")

# Load the model
model = load_model('my_model.h5')

# Name of Classes
target_names = ["Common_Rust", "Gray_Leaf_Spot", "Healthy", "Blight"]

disease_explanations = {
    "Common_Rust": "Common Rust is a fungal disease affecting plants, characterized by orange pustules on leaves.",
    "Gray_Leaf_Spot": "Gray Leaf Spot is a fungal disease affecting corn plants, causing small, rectangular lesions on leaves.",
    "Healthy": "No disease detected. The plant appears to be healthy.",
    "Blight": "Blight is a fungal disease affecting various crops, causing rapid wilting and browning of foliage."
}

# Dictionary mapping diseases to insects
disease_insects = {
    "Common_Rust": ["Corn Earworm", "Aphids", "Spider Mites", "Thrips", "Fall Armyworm", "Japanese Beetle",
                    "Leafhoppers", "Wireworms", "Grasshoppers", "Rice Stink Bug"],
    "Gray_Leaf_Spot": ["Corn Borers", "Corn Leaf Aphids", "European Corn Borer", "Northern Corn Leaf Blight",
                       "Southern Corn Leaf Blight", "Stalk Borers", "Grasshoppers", "Rice Stink Bug", "Sugarcane Borer",
                       "Rice Leaf Miner"],
    "Blight": ["Potato Aphids", "Whiteflies", "Colorado Potato Beetle", "Tomato Hornworm", "Cabbage Loopers",
               "Onion Thrips", "Flea Beetles", "Raspberry Beetle", "Cutworms", "Squash Bug"]
}

# Dictionary mapping diseases to treatments
disease_treatments = {
    "Common_Rust": ["Fungicides", "Crop Rotation", "Resistant Varieties", "Planting Date Adjustments",
                    "Removing Infected Plant Material"],
    "Gray_Leaf_Spot": ["Fungicides", "Tillage Practices", "Crop Rotation", "Planting Date Adjustments",
                       "Fertility Management"],
    "Blight": ["Fungicides", "Resistant Varieties", "Planting Date Adjustments", "Removing Infected Plant Material",
               "Sanitation Practices"]
}

# Dictionary mapping diseases to preventions
disease_preventions = {
    "Common_Rust": ["Crop Rotation", "Resistant Varieties", "Good Weed Control", "Sanitation Practices",
                    "Monitoring and Early Detection"],
    "Gray_Leaf_Spot": ["Crop Rotation", "Resistant Varieties", "Good Fertility Management", "Weed Control",
                       "Monitoring and Early Detection"],
    "Blight": ["Crop Rotation", "Resistant Varieties", "Good Drainage", "Sanitation Practices",
               "Monitoring and Early Detection"]
}

app = FastAPI(
    title="Plant Disease Detection API",
    description="""An API that utilizes a Deep Learning model built with Keras(Tensorflow) to detect if a plant is healthy or suffering from Rust and Powder formation.""",
    version="0.0.1",
    debug=True,
)

Api_Key = "2606f769271b8d545fe3458b2b72ed9f"  # Paste Your API ID Here

model_pest_crop = MobileNetV2(weights='imagenet')

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def predict_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model_pest_crop.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds

# Function to create a new database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Query the database for the user with the provided email
    user = db.query(User).filter(User.email == email).first()

    # Check if user exists and password is correct
    if not user or not pwd_context.verify(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return token and user object
    return {"access_token": generate_random_token(), "user": user}

@app.post("/predict/disease")
async def root(file: UploadFile = File(...)):
    file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize the image to 256x256
    img = cv2.resize(img, (299, 299))

    # Add a batch dimension to the image
    img = np.expand_dims(img, axis=0)

    # Get the prediction probabilities from the model
    prediction_probabilities = model.predict(img)[0]

    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction_probabilities)

    # Get the confidence level (probability) of the prediction
    confidence_level = prediction_probabilities[predicted_class_index]

    # Get the class label corresponding to the predicted index
    predicted_class_label = target_names[predicted_class_index]

    # Get the list of insects associated with the predicted disease
    insects = disease_insects.get(predicted_class_label, [])

    # Get the list of treatments associated with the predicted disease
    treatments = disease_treatments.get(predicted_class_label, [])

    # Get the list of preventions associated with the predicted disease
    preventions = disease_preventions.get(predicted_class_label, [])

    # Get the brief explanation of the predicted disease
    disease_explanation = disease_explanations.get(predicted_class_label, "No explanation available.")

    result = {
        "prediction": predicted_class_label,
        "confidence_level": float(confidence_level),
        "disease_explanation": disease_explanation,
        "associated_insects": insects,
        "treatments": treatments,
        "preventions": preventions
    }
    return result


@app.post("/push-message")
async def push_message(title: str, message: str, db: Session = Depends(get_db)):
    pusher_client = pusher.Pusher(
        app_id='1663724',
        key='a7d67608f236b9565325',
        secret='b83f97c2ef83e7d7dbbf',
        cluster='eu',
        ssl=True
    )
    # Trigger Pusher event for each user
    users = db.query(User).all()
    pusher_client.trigger('my-channel', 'my-event', {'message': title})

    # Store message in database for each user
    for user in users:
        new_notification = Notification(title=title, message=message, user_id=user.id)
        db.add(new_notification)
    db.commit()

    return {"message": "Message pushed successfully and stored in the database"}

@app.post("/identify/crop/pest")
async def identify_crop_disease(file: UploadFile = File(...)):
    image_data = await file.read()
    predictions = predict_image(image_data)
    response = []
    for i, (_, label, score) in enumerate(predictions):
        response.append({"label": label, "score": float(score)})  # Convert score to float
    return response




# SQLAlchemy database URL
DATABASE_URL = "postgresql://postgres:allpha01@localhost/postgres"

# Create a database instance
database = databases.Database(DATABASE_URL)

# Create a SQLAlchemy engine
engine = sqlalchemy.create_engine(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define a base class for ORM models
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Notification(Base):
    __tablename__ = 'notifications'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    message = Column(String)
    seen = Column(Boolean, default=False)  # Add the 'seen' field
    disable = Column(Boolean, default=False)  # Add the 'seen' field
    created_at = Column(DateTime, server_default=func.now())

    # Define foreign key relationship with User table
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User", back_populates="notifications")

class UserQuery(Base):
    __tablename__ = 'queries'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String)
    message = Column(String)
    name = Column(String)
    email = Column(String)
    disable = Column(Boolean, default=False)  # Add the 'seen' field

    # Define relationship with the User table
    user = relationship("User", back_populates="queries")

# Define SQLAlchemy model for User
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    disable = Column(Boolean, default=False)  # Add the 'seen' field
    phone = Column(String)
    password = Column(String)
    role = Column(String, default="user")

    # Define relationship with notifications
    notifications = relationship("Notification", back_populates="user")
    queries = relationship("UserQuery", back_populates="user")


# Define SQLAlchemy model for MarketData
class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    admin1 = Column(String)
    admin2 = Column(String)
    market = Column(String)
    latitude = Column(String)
    longitude = Column(String)
    category = Column(String)
    commodity = Column(String)
    unit = Column(String)
    priceflag = Column(String)
    pricetype = Column(String)
    currency = Column(String)
    price = Column(String)
    usdprice = Column(String)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Create a sessionmaker instance
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create a new database sessionu
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
    disable: bool = False  # Default value for the 'disable' field
    phone: str = None  # Default value for the 'phone' field (assuming it's optional)

class CSVData(BaseModel):
    csv_data: str

# Tkinter app for file selection and CSV upload
class UploadApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Upload CSV")
        self.root.geometry("300x150")

        self.label = tk.Label(self.root, text="Select CSV file:")
        self.label.pack()

        self.upload_button = tk.Button(self.root, text="Browse", command=self.upload_csv)
        self.upload_button.pack()

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                with open(file_path, newline='') as file:
                    csv_data = file.read()

                # Process CSV data
                self.process_csv(csv_data)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def process_csv(self, csv_data):
        # Parse CSV data
        csv_data = csv_data.strip()
        csv_file = StringIO(csv_data)
        reader = csv.DictReader(csv_file)

        # Create a list to store MarketData objects
        market_data_list = []

        # Iterate through CSV rows and create MarketData objects
        for row in reader:
            market_data = MarketData(
                date=row['date'],
                admin1=row['admin1'],
                admin2=row['admin2'],
                market=row['market'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                category=row['category'],
                commodity=row['commodity'],
                unit=row['unit'],
                priceflag=row['priceflag'],
                pricetype=row['pricetype'],
                currency=row['currency'],
                price=row['price'],
                usdprice=row['usdprice']
            )
            market_data_list.append(market_data)

        # Add the market data to the database
        with SessionLocal() as db:
            db.add_all(market_data_list)
            db.commit()

        messagebox.showinfo("Success", "CSV data uploaded successfully")

    def run(self):
        self.root.mainloop()

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
        disable=user_data.disable,
        phone=user_data.phone,
        role=user_data.role
    )

    # Add the new user to the session and commit the transaction
    db.add(new_user)
    db.commit()

    return {"user_id": new_user.id}

@app.get("/get_user/queries/{user_id}")
def get_user_queries(user_id: int, db: Session = Depends(get_db)):
    # Query the database to retrieve all queries
    user_queries = db.query(UserQuery).filter(UserQuery.user_id == user_id).all()

    # Check if user exists
    if not user_queries:
        raise HTTPException(status_code=404, detail="User not found")
    # queries = db.query(Query).all()
    return user_queries

@app.get("/all/queries")
async def get_all_queries(db: Session = Depends(get_db)):
    # Query the database to retrieve all users
    queries = db.query(UserQuery).all()
    return queries

@app.post("/send/queries/{user_id}/")
def create_query(title: str, message: str, user_id: int, name: str, email: str, db: Session = Depends(get_db)):
    # Create a new query in the database
    new_query = UserQuery(user_id=user_id, title=title, message=message, name=name, email=email)
    db.add(new_query)
    db.commit()
    db.refresh(new_query)
    return new_query

@app.get("/notifications/user/{user_id}")
async def get_notifications(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    notifications = db.query(Notification).filter(Notification.user_id == user_id).all()
    return notifications

@app.get("/notification/{notification_id}")
async def get_notification(notification_id: int, db: Session = Depends(get_db)):
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    # Update seen field to True
    notification.seen = True
    db.commit()
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification

@app.get("/disable/notification/{notification_id}")
async def get_notification(notification_id: int, db: Session = Depends(get_db)):
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    # Update seen field to True
    notification.disable = True
    db.commit()
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification

@app.get("/users")
async def get_all_users(db: Session = Depends(get_db)):
    # Query the database to retrieve all users
    users = db.query(User).all()
    return users

@app.delete("/user/disable/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    # Query the database to retrieve the user by their ID
    user = db.query(User).filter(User.id == user_id).first()

    # If user is None, raise an HTTPException with status code 404
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Toggle the value of the 'disable' field
    user.disable = not user.disable

    # Commit the changes to the database
    db.commit()
    user = db.query(User).filter(User.id == user_id).first()

    return {"message": "User disabled", "user_id": user.id, "disable": user.disable}


@app.get("/user/{user_id}")
async def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
    # Retrieve user by user ID
    user = db.query(User).filter(User.id == user_id).first()

    # Check if user exists
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user

@app.get("/market_prices/{market_price_id}")
async def get_market_price_by_id(market_price_id: int, db: Session = Depends(get_db)):
    # Query the database to retrieve the market price by its ID
    market_price = db.query(MarketData).filter(MarketData.id == market_price_id).first()

    # If market_price is None, raise an HTTPException with status code 404
    if market_price is None:
        raise HTTPException(status_code=404, detail="Market price not found")

    return market_price

@app.get("/market/prices/paginate")
async def paginate_market_prices(page: int = Query(default=1, ge=1), db: Session = Depends(get_db)):
    # Execute the query to get all market prices
    market_prices = db.query(MarketData).all()

    # Calculate total number of pages
    total_pages = (len(market_prices) + 5) // 6

    # Calculate start and end index for pagination
    start_index = len(market_prices) - (page * 6)
    end_index = max(start_index - 6, 0)

    # Paginate the market prices
    paginated_market_prices = market_prices[start_index:end_index:-1]

    return {
        "page": page,
        "total_pages": total_pages,
        "market_prices": paginated_market_prices
    }

@app.get("/market_prices/")
async def search_market_prices(admin1: str = None, admin2: str = None, commodity: str = None, page: int = Query(default=1, ge=1), db: Session = Depends(get_db)):
    # Initialize query filters
    filters = []

    # Generate all combinations of admin1, admin2, and commodity
    for a1 in [admin1, None]:
        for a2 in [admin2, None]:
            for comm in [commodity, None]:
                # Create a filter for the current combination
                filter_combination = []
                if a1:
                    filter_combination.append(MarketData.admin1 == a1)
                if a2:
                    filter_combination.append(MarketData.admin2 == a2)
                if comm:
                    filter_combination.append(MarketData.commodity == comm)

                # Add the filter combination to the filters list
                if filter_combination:
                    filters.append(filter_combination)

    # Execute the query with applied filters
    market_prices = []
    for f in filters:
        query = db.query(MarketData).filter(*f).all()
        market_prices.extend(query)

    total_pages = (len(market_prices) + 5) // 6

    # Calculate start and end index for pagination
    start_index = len(market_prices) - (page * 6)
    end_index = max(start_index - 6, 0)

    # Paginate the market prices
    paginated_market_prices = market_prices[start_index:end_index:-1]

    return {
        "page": page,
        "total_pages": total_pages,
        "market_prices": paginated_market_prices
    }

# This modified code will provide you with the weather forecast for
# the next 5 days, grouped by day. Each day will have a list of forecast
# entries containing weather condition, temperature, pressure, humidity, and wind speed.

@app.get("/weather")
async def get_weather(city: str):
    apiUrl = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={Api_Key}"

    json_data = requests.get(apiUrl).json()

    forecast_data = json_data['list']  # List of forecast data for the next few days

    # Initialize empty lists to store forecast data for each day
    forecast_by_day = [[] for _ in range(5)]

    # Iterate through forecast data and group it by day
    for data in forecast_data:
        timestamp = data['dt']  # Timestamp of forecast data
        date = datetime.fromtimestamp(timestamp)  # Convert timestamp to datetime
        day_index = (date - datetime.now()).days  # Calculate day index relative to today

        # Extract relevant weather information
        condition = data['weather'][0]['main']
        temp = int(data['main']['temp'] - 273.15)
        min_temp = int(data['main']['temp_min'] - 273.15)
        max_temp = int(data['main']['temp_max'] - 273.15)
        pressure = data['main']['pressure']
        humidity = data['main']['humidity']
        wind = data['wind']['speed']

        # Append forecast data to the corresponding day's list
        forecast_by_day[day_index].append({
            "Time": date.strftime('%Y-%m-%d %H:%M:%S'),
            "Weather Condition": condition,
            "Temperature": f"{temp}°C",
            "Min Temperature": f"{min_temp}°C",
            "Max Temperature": f"{max_temp}°C",
            "Pressure": pressure,
            "Humidity": f"{humidity}%",
            "Wind Speed": f"{wind} m/s"
        })

    return forecast_by_day

@app.get("/weather/paginate")
async def get_weather(city: str, day: int = 0):
    apiUrl = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={Api_Key}"

    json_data = requests.get(apiUrl).json()

    forecast_data = json_data['list']  # List of forecast data for the next few days

    # Initialize empty list to store forecast data for the requested day
    forecast_for_requested_day = []

    # Calculate the date for the requested day
    requested_date = (datetime.now() + timedelta(days=day)).date()

    icon_base_url = "http://openweathermap.org/img/wn/"

    # Iterate through forecast data and filter it for the requested day
    for data in forecast_data:
        timestamp = data['dt']  # Timestamp of forecast data
        date = datetime.fromtimestamp(timestamp).date()  # Convert timestamp to date

        # Check if the forecast data corresponds to the requested day
        if date == requested_date:
            # Extract relevant weather information
            condition = data['weather'][0]['main']
            temp = int(data['main']['temp'] - 273.15)
            min_temp = int(data['main']['temp_min'] - 273.15)
            max_temp = int(data['main']['temp_max'] - 273.15)
            pressure = data['main']['pressure']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            icon_code = data['weather'][0]['icon']  # Get weather icon code

            # Build icon URL
            icon_url = f"{icon_base_url}{icon_code}@2x.png"

            # Append forecast data to the list
            forecast_for_requested_day.append({
                "date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                "condition": condition,
                "temp": f"{temp}°C",
                "min_temp": f"{min_temp}°C",
                "max_temp": f"{max_temp}°C",
                "pressure": pressure,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind} m/s",
                "icon": icon_url  # Include icon URL in response
            })

    return forecast_for_requested_day

if __name__ == "__main__":
    # message = input("Enter the message to send: ")
    # publish_message(message)
    upload_app = UploadApp()
    upload_app.run()

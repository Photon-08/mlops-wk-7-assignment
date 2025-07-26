from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("model.joblib")

# Define request body format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Iris classification API is live for prediction!"}

@app.post("/predict")
def predict_iris(data: IrisInput):
    # Convert the input into a 2D array for model
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    
    return {"prediction": prediction[0]}

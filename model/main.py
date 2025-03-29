from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# ---------------------------
# Create FastAPI Instance
# ---------------------------
app = FastAPI()

# ---------------------------
# Load the Saved Model
# ---------------------------
# Make sure that the file 'autism_behavior_model.pkl' is in the same directory as this script.
model = joblib.load("autism_behavior_model.pkl")

# ---------------------------
# Define Mappings for Categorical Features
# ---------------------------
# These mappings must match the encoding used during training.
# For example, if LabelEncoder was used on the 'Gender' column, it likely sorted alphabetically.
# In this example, assume:
#   Gender: "Female" -> 0, "Male" -> 1
#   Current Mood: "Anxious" -> 0, "Frustrated" -> 1, "Happy" -> 2, "Neutral" -> 3, "Sad" -> 4
gender_mapping = {"Female": 0, "Male": 1}
current_mood_mapping = {
    "Anxious": 0,
    "Frustrated": 1,
    "Happy": 2,
    "Neutral": 3,
    "Sad": 4
}

# Inverse mapping for the target "Level"
# Assuming the Level LabelEncoder sorted the classes lexicographically, for example:
#   0: "High", 1: "Low", 2: "Moderate", 3: "Very High", 4: "Very Low"
# Adjust these as needed based on your training.
level_inverse_mapping = {
    0: "High",
    1: "Low",
    2: "Moderate",
    3: "Very High",
    4: "Very Low"
}

# ---------------------------
# Define the Request Body using Pydantic
# ---------------------------
class PredictionRequest(BaseModel):
    Age: int
    Gender: str             # e.g., "Female" or "Male"
    Current_Mood: str       # e.g., "Happy", "Neutral", etc.
    Parent_Satisfaction: int
    Engagement_Level: int
    Completed_Tasks: int
    Time_Spent: float
    Correct_in_First_Attempt: int

# ---------------------------
# Define the Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert incoming data to a dictionary
        input_data = request.dict()
        
        # Check that the categorical string values are valid
        if input_data["Gender"] not in gender_mapping:
            raise HTTPException(status_code=400, detail="Invalid Gender value.")
        if input_data["Current_Mood"] not in current_mood_mapping:
            raise HTTPException(status_code=400, detail="Invalid Current_Mood value.")
        
        # Create a DataFrame from the request data, mapping the string labels to numeric codes.
        df = pd.DataFrame([{
            "Age": input_data["Age"],
            "Gender": gender_mapping[input_data["Gender"]],
            "Current Mood": current_mood_mapping[input_data["Current_Mood"]],
            "Parent Satisfaction": input_data["Parent_Satisfaction"],
            "Engagement Level": input_data["Engagement_Level"],
            "Completed Tasks": input_data["Completed_Tasks"],
            "Time Spent": input_data["Time_Spent"],
            "Correct in First Attempt": input_data["Correct_in_First_Attempt"]
        }])
        
        # Use the loaded model to predict the numeric code for 'Level'
        pred_numeric = model.predict(df)[0]
        
        # Convert the numeric prediction back to a string label using the inverse mapping
        pred_label = level_inverse_mapping.get(pred_numeric, "Unknown")
        
        return {"prediction": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

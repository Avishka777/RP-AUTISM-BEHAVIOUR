from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import joblib
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

# ---------------------------
# Create FastAPI Instance
# ---------------------------
app = FastAPI()

# ---------------------------
# Load the Saved Autism Behavior Model
# ---------------------------
# Ensure the file 'autism_behavior_model.pkl' is in the same directory as this script.
autism_model = joblib.load("autism_behavior_model.pkl")

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
# Define the Request Body using Pydantic for Prediction
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
# Prediction Endpoint for Autism Behavior Model
# ---------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.dict()

        # Validate categorical inputs
        if input_data["Gender"] not in gender_mapping:
            raise HTTPException(status_code=400, detail="Invalid Gender value.")
        if input_data["Current_Mood"] not in current_mood_mapping:
            raise HTTPException(status_code=400, detail="Invalid Current_Mood value.")

        # Create a DataFrame from input data with proper mappings
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

        # Predict numeric code using the loaded model
        pred_numeric = autism_model.predict(df)[0]

        # Map numeric prediction back to string label
        pred_label = level_inverse_mapping.get(pred_numeric, "Unknown")

        return {"prediction": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Load the YOLOv8 Model for Object Detection
# ---------------------------
yolo_model = YOLO("yolov8n.pt")

# ---------------------------
# Object Detection Endpoint using YOLOv8
# ---------------------------
@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Run YOLOv8 object detection
        results = yolo_model(image)

        detections = []
        for result in results:
            for box in result.boxes:
                obj = {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                }
                detections.append(obj)

        return {"detected_objects": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

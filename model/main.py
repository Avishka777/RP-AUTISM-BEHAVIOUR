from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import joblib
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tempfile
from typing import List, Dict, Optional
from catboost import CatBoostClassifier

# ---------------------------
# Create FastAPI Instance
# ---------------------------
app = FastAPI()

# ---------------------------
# Load Models
# ---------------------------
# Load the CatBoost model
autism_model = CatBoostClassifier()
autism_model.load_model("models/autism_behavior_catboost_model.cbm") 

# Load the encoders used during training
encoders = joblib.load("models/autism_behavior_encoders.pkl")  

yolo_model = YOLO("models/yolov8n.pt")
emotion_model = load_model("models/best_emotion_model.h5")

# ---------------------------
# Emotion Class Mapping
# ---------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------------------
# Define Mappings for Categorical Features
# ---------------------------
# Define valid values that the API will accept
valid_genders = ["0", "1"]  # API accepts "0" or "1" as strings
valid_moods = list(encoders["Current Mood"].classes_)

# Create mapping from API input to encoded values
gender_encoding = {"0": 0, "1": 1}  # Maps API input to model expected input

# Create mapping from encoded values back to labels for response
gender_labels = {0: "Male", 1: "Female"}  # For displaying in responses
level_inverse_mapping = {i: cls for i, cls in enumerate(encoders["Level"].classes_)}

# ---------------------------
# Prediction Request Model
# ---------------------------
class ActivityEntry(BaseModel):
    completed: bool
    timeSpent: int
    marks: int
    parentSatisfaction: Optional[int] = None

class PredictionRequest(BaseModel):
    Age: int
    Gender: str
    Current_Mood: str
    activities: Dict[str, ActivityEntry]

# ---------------------------
# Calculation function
# ---------------------------
def calc(data: dict) -> dict:
    """
    Compute all derived fields for the /predict endpoint.
    Expects data to contain:
      - "Age": int
      - "Gender": str
      - "Current_Mood": str
      - "activities": dict of activity_id -> {
            "completed": bool,
            "timeSpent": int,
            "marks": int,
            "parentSatisfaction": Optional[int]
        }
    Returns a dict with the exact inputs your model expects:
      {
        "Age": int,
        "Gender": str,
        "Current_Mood": str,
        "Parent_Satisfaction": int,
        "Engagement_Level": int,
        "Completed_Tasks": int,
        "Time_Spent": int,
        "Correct_in_First_Attempt": int
      }
    """
    
    activities = data.get("activities", {})
    total_activities = len(activities)

    # 1) Parent satisfaction: average over activity1, activity2, activity3_5
    ps_keys = ["activity1", "activity2", "activity3_5"]
    ps_values = [
        activities[k].get("parentSatisfaction", 0) for k in ps_keys
        if k in activities
    ]
    # avoid division by zero; assume 3 keys always present
    avg_parent_satisfaction = round(sum(ps_values) / len(ps_values)) if ps_values else 0

    # 2) Completed tasks and %
    completed_flags = [act.get("completed", False) for act in activities.values()]
    completed_count = sum(1 for f in completed_flags if f)
    completed_pct = (completed_count / total_activities) * 100 if total_activities else 0
    completed_val = completed_pct / 10 if completed_pct else 0

    # 3) Correct-in-first-attempt tasks and %
    #    (marks > 0 counts as correct)
    correct_count = sum(1 for act in activities.values() if act.get("marks", 0) > 0)

    # 4) Total time spent on completed tasks
    total_time_spent = sum(
        act.get("timeSpent", 0) for act in activities.values() if act.get("completed", False)
    )

    # 5) Average time per completed task
    avg_time = (total_time_spent / completed_count) if completed_count else 0

    # 6) Raw engagement score = average of [avg_time, completed_pct, correct_pct]
    correct_pct = (correct_count / total_activities) * 100 if total_activities else 0
    raw_engagement = (avg_time + completed_pct + correct_pct) / 3
    correct_val = correct_pct / 10 if correct_pct else 0

    # 7) Scale raw_engagement (0–100) into a 1–5 integer range
    #    (simply dividing by 20, then clamping & rounding to [1,5])
    engagement_level = int(min(max(raw_engagement / 20, 1), 5))

    x = {
        "Age": data.get("Age", 0),
        "Gender": data.get("Gender", ""),
        "Current_Mood": data.get("Current_Mood", ""),
        "Parent_Satisfaction": avg_parent_satisfaction,
        "Engagement_Level": engagement_level,
        "Completed_Tasks": completed_val,
        "Time_Spent": total_time_spent,
        "Correct_in_First_Attempt": correct_val,
    }
    
    print(x)
    return x

# ---------------------------
# Prediction Endpoint for Autism Behavior Model
# ---------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        raw = request.dict()
        input_data = calc(raw)  # Use this function to perform calculations

        # Convert string gender to the expected format
        if input_data["Gender"].lower() in ["male", "m"]:
            input_data["Gender"] = "0"
        elif input_data["Gender"].lower() in ["female", "f"]:
            input_data["Gender"] = "1"

        # Validate categorical inputs
        if input_data["Gender"] not in valid_genders:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Gender value", 
                    "valid_values": valid_genders,
                    "note": "Use '0' for Male, '1' for Female or send 'Male'/'Female'"
                }
            )
        
        if input_data["Current_Mood"] not in valid_moods:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Current_Mood value", 
                    "valid_values": valid_moods
                }
            )
        
        # Additional validations for numerical inputs
        # if not (1 <= input_data["Parent_Satisfaction"] <= 5):
        #     raise HTTPException(
        #         status_code=400,
        #         detail={"error": "Invalid Parent_Satisfaction value", "valid_range": "1 to 5"}
        #     )
        
        if not (1 <= input_data["Completed_Tasks"] <= 15):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Completed_Tasks value", "valid_range": "1 to 15"}
            )
        
        if not (6 <= input_data["Age"] <= 10):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Age value", "valid_range": "6 to 10"}
            )
        
        if input_data["Time_Spent"] <= 0:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Time_Spent value", "valid_value": "greater than 0"}
            )

        # Create a DataFrame from input data with proper encoding
        df = pd.DataFrame([{
            "Age": input_data["Age"],
            "Gender": gender_encoding[input_data["Gender"]],  # Convert to model's expected encoding
            "Current Mood": encoders["Current Mood"].transform([input_data["Current_Mood"]])[0],
            "Parent Satisfaction": input_data["Parent_Satisfaction"],
            "Engagement Level": input_data["Engagement_Level"],
            "Completed Tasks": input_data["Completed_Tasks"],
            "Time Spent": input_data["Time_Spent"],
            "Correct in First Attempt": input_data["Correct_in_First_Attempt"]
        }])

        # Predict numeric code using the loaded CatBoost model
        pred_numeric = autism_model.predict(df)[0]
        
        # Get probabilities for each class
        pred_probabilities = autism_model.predict_proba(df)[0]
        
        # Map numeric prediction back to string label
        pred_label = encoders["Level"].inverse_transform([pred_numeric])[0]

        # Define suggestions for each prediction level
        suggestions_dict = {
            "Very Low": [
                "අන්තර්ක්‍රියාත්මක සහ උත්තේජක ක්‍රියාකාරකම් හඳුන්වා දීමෙන් අවදාන මට්ටම වැඩි කරන්න.",
                " සාර්ථකත්වය වැඩි කිරීමට කෙටි, අවධානය යොමු කළ කාර්යයන් ලබා දෙන්න."
                "දරුවා සතුටින් සිටින මොහොතක පාඩම් සදහා යොමු වන්න"
                "ධනාත්මක ප්‍රතිචාර වැඩි කරමින් සහ සුළු ජයග්‍රහණයන් හඳුනා ගැනීමෙන් මනෝභාවය වැඩි කරන්න.",
                "මාපියන්ගේ සහභාගීත්වය වැඩි කරමින් සතුට වැඩි කරනු සහ ළමයාට නිතර සහයෝගය ලැබීම සුරක්ෂිත කරන්න."

            ],
            "Low": [
                "දෘශ්‍ය උදව් සහ ප්‍රසාද ලබා දීමෙන් උද්යෝගය සහ ඇලීම වැඩි කරන්න."
                "දරුවාට පාඩම එදිනදා උදාහරන සහිතව නිරතුරුව ආවර්ජනය කරන්න"
                "ඇලීම තබා ගැනීමට සහ නිරාශාව අඩු කිරීමට ළමයාගේ කාර්ය සාධනය අනුව කාර්යයන්ගේ සංකීර්ණතාව සකසන්න.",
                "ළමයාගේ හැඟීම් පිළිගනිමින් සහ සන්සුන් වීමට උපකාරී විධාන ලබා දීමෙන් මානසික සහයෝගය ලබා දෙන්න.",
                "කාර්ය නිම කිරීම සඳහා ළමයාගේ අවශ්‍යතා තේරුම් ගැනීමට සහ වැඩිදියුණු කිරීමට දෙමාපියන් සමඟ සන්නිවේදනය වැඩි කරන්න."
            ],
            "Moderate": [
                "වර්ධනය සහ විශ්වාසය සඳහා ළමයාගේ වර්තමාන කුසලතා මට්ටමට වඩා තරමක් සඔකීර්න හඳුන්වා දෙන්න.",
                "ඇලීම සහ ගැටළු විසඳීමේ කුසලතා වැඩි කිරීමට කණ්ඩායම් කටයුතු හෝ සමවින්දනය එක් කරන්න.",
                "වැඩිදියුණු කිරීමට තවත් උනන්දු කිරීම සඳහා ළමයාගේ කාර්ය සාධනය පිළිබඳව කාලෝචිත ප්‍රතිචාර ලබා දෙන්න.",
                "නිරාශාව පාලනය කරමින් අවධානය තබා ගැනීමට ස්වයං පාලන තාක්‍ෂණ උද්යෝග කරන්න."
                "එදිනදා ජීවිතයේදී හැසිරීම සම්බන්ද අවස්තාවලදී දරුවාගෙන් ප්‍රශ්න කර පිලිතුරු ලබාගන්න"
                "හැකි සෑමවිටම හැසිරීම සම්බන්ද හොද පුරුදු නිවස තුල බාවිතා කරන්න"

            ],
            "High": [
                "වර්ධනය සහ දක්ෂතාව වැඩි කිරීමට ළමයාගේ වර්තමාන හැකියාවන් අභියෝගයට ලක් කරන වඩා සංකීර්ණ කාර්යයන් ලබා දෙන්න.",
                "විශ්වාසය සහ ස්වාධීනභාවය ගොඩනගීමට ස්වාධීන ගැටළු විසඳීම සහ තීරණ ගැනීම උද්යෝග කරන්න.",
                "ඇලීම පවත්වාගෙන යාමට සහ මනෝභාවය තබා ගැනීමට ධනාත්මක ප්‍රතිචාර භාවිතා කරන්න.",
                " සංවිධාන කුසලතා වැඩි කිරීමට ළමයාට අනෙකුත් අය ගුරුවරයෙක් ලෙස උපදේශනය කිරීමට අවස්ථා ලබා දෙන්න."
                "දරුවා අනෙක් ලමුන් සමග මුහු කරමින් ප්‍ර්‍රායෝගික  අත්දැකීම් විදීමට උපකාර කරන්න"
                "ඔබ දරුවා සමග සම්බන්ද වෙමින් පාඩමට අදාල අවස්තා නිවස තුල ගොඩ නගන්න"
            ],
            "Very High": [
                "දරුවාට හැහි සෑම විටම ප්‍රයෝගිකව පරිසරය සකසා දෙන්න"
                "හැසිරීම සම්බන්ද අපගේ අනෙක් පැවරුම් ලබා ගන්න"
                "දරුවා සාමන්‍ය දරුවන් සමග මුහු කරන්න උත්සාහ ගන්න"
                "ළමයාගේ සම්පූර්ණ හැකියාවන් පෙන්වීමට ඉඩ දෙන උසස් කාර්යයන් සහ ව්‍යාපෘති හඳුන්වා දෙන්න.",
                "කණ්ඩායම් කාර්යය නියෝජනය කිරීම හෝ සහෝදරයන්ට මඟ පෙන්වීම වැනි නේතෘත්ව අවස්ථා ලබා දෙන්න.",
                "හැකි සෑම විටම දරුවා පොදු ස්තාන වලට රැගෙන යන්න.",
                "අනාගත සාර්ථකතා සහ රැකියාවන්ට අවධානය යොමු කිරීමට ස්වයං පරිකල්පනය සහ අරමුණු නියම කිරීම උද්යෝග කරන්න."
            ]
        }

        suggestions = suggestions_dict.get(pred_label, [])
        
        # Return the response in the expected format
        return {
            "prediction": pred_label,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# ---------------------------
# Load the YOLOv8 Model for Object Detection
# ---------------------------
yolo_model = YOLO("models/yolov8n.pt")

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
# Emotion Detection Endpoint
# ---------------------------
@app.post("/detect_emotion/")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((48, 48))

        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 48, 48, 1)

        prediction = emotion_model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[predicted_index]
        confidence = float(np.max(prediction))

        return {"emotion": predicted_emotion, "confidence": confidence}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/detect_emotion_video/")
async def detect_emotion_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        # Process video frames
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frames to process (first 5 seconds)
        target_frames = 15
        frames_to_process = min(int(fps * 5), total_frames)  # Max 5 seconds worth of frames
        frame_interval = max(1, frames_to_process // target_frames)  # Ensure we get exactly 15 frames
        
        emotions = []
        processed_frames = 0
        current_frame = 0

        while cap.isOpened() and processed_frames < target_frames and current_frame < frames_to_process:
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break

            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Convert to grayscale and resize
            gray_image = pil_image.convert("L").resize((48, 48))
            img_array = img_to_array(gray_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict emotion
            prediction = emotion_model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[predicted_index]
            confidence = float(np.max(prediction))

            emotions.append({
                "frame": current_frame,
                "time": current_frame / fps,  # Time in seconds
                "emotion": predicted_emotion,
                "confidence": confidence
            })

            processed_frames += 1
            current_frame += frame_interval

        cap.release()
        
        # Calculate emotion statistics
        emotion_stats = {}
        for e in emotion_labels:
            count = sum(1 for x in emotions if x["emotion"] == e)
            emotion_stats[e] = {
                "count": count,
                "percentage": (count / len(emotions)) * 100 if emotions else 0
            }

        # Get dominant emotion
        dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1]["percentage"])[0] if emotions else None

        return {
            "video_info": {
                "original_duration": duration,
                "processed_duration": min(5.0, duration),
                "original_fps": fps,
                "frames_analyzed": len(emotions)
            },
            "emotion_percentages": {k: v["percentage"] for k, v in emotion_stats.items()},
            "dominant_emotion": dominant_emotion,
        }

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        # Clean up temporary file
        if 'temp_video_path' in locals():
            import os
            os.unlink(temp_video_path)

# ---------------------------
# Run Server (for local testing)
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
from typing import List
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
class PredictionRequest(BaseModel):
    Age: int 
    Gender: str  # Will accept "0" or "1" as strings
    Current_Mood: str
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
        if input_data["Gender"] not in valid_genders:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Gender value", 
                    "valid_values": valid_genders,
                    "note": "Use '0' for Male, '1' for Female"
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
        if not (1 <= input_data["Parent_Satisfaction"] <= 5):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Parent_Satisfaction value", "valid_range": "1 to 5"}
            )
        
        if not (1 <= input_data["Completed_Tasks"] <= 10):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Completed_Tasks value", "valid_range": "1 to 10"}
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
                "Increase the engagement level by introducing interactive and stimulating activities.",
                "Offer shorter, more focused tasks to prevent frustration and increase the chances of success.",
                "Improve mood by providing more positive reinforcement and recognizing small achievements.",
                "Enhance parental involvement to boost satisfaction and ensure more consistent support for the child."
            ],
            "Low": [
                "Provide visual aids and rewards to enhance motivation and engagement in tasks.",
                "Adjust the complexity of tasks based on the child's performance to maintain interest and reduce frustration.",
                "Offer emotional support by recognizing the child's feelings and providing calming strategies.",
                "Increase communication with parents to better understand the child's needs and improve task completion."
            ],
            "Moderate": [
                "Introduce challenges that are slightly above the child's current skill level to promote growth and confidence.",
                "Incorporate teamwork or peer interactions to boost engagement and problem-solving skills.",
                "Ensure that the child receives timely feedback on performance to encourage further improvement.",
                "Encourage self-regulation techniques to help manage frustration and maintain focus."
            ],
            "High": [
                "Provide more complex tasks that challenge the child's current abilities to foster growth and mastery.",
                "Encourage independent problem-solving and decision-making to build confidence and autonomy.",
                "Use positive reinforcement to sustain motivation and recognize progress toward mastery.",
                "Offer opportunities for the child to mentor others, which could enhance leadership and organizational skills."
            ],
            "Very High": [
                "Introduce advanced tasks and projects that allow the child to demonstrate their full capabilities.",
                "Provide leadership opportunities, such as managing a group task or guiding peers in activities.",
                "Offer opportunities for skill development in a specialized area (e.g., music, art, or technology) to foster expertise.",
                "Promote self-reflection and goal-setting to help the child focus on future achievements and career paths."
            ]
        }

        suggestions = suggestions_dict.get(pred_label, [])
        
        # Create probability dictionary
        probability_dict = {
            level: float(pred_probabilities[i]) 
            for i, level in enumerate(encoders["Level"].classes_)
        }
        
        # Include the gender label in the response for clarity
        gender_label = gender_labels[gender_encoding[input_data["Gender"]]]
        
        return {
            "prediction": pred_label,
            "probabilities": probability_dict,
            "suggestions": suggestions,
            "input_gender": {
                "received_value": input_data["Gender"],
                "interpreted_as": gender_label
            }
        }
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
        # Validate file is a video
        # if not file.content_type.startswith('video/'):
        #     raise HTTPException(400, "File must be a video")

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

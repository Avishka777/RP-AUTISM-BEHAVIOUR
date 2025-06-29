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

# ---------------------------
# Create FastAPI Instance
# ---------------------------
app = FastAPI()

# ---------------------------
# Load Models
# ---------------------------
autism_model = joblib.load("models/autism_behavior_model.pkl")
yolo_model = YOLO("models/yolov8n.pt")
emotion_model = load_model("models/best_emotion_model.h5")

# ---------------------------
# Emotion Class Mapping
# ---------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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
# Prediction Request Model
# ---------------------------
class PredictionRequest(BaseModel):
    Age: int 
    Gender: str
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
        if input_data["Gender"] not in gender_mapping:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Gender value", "valid_values": list(gender_mapping.keys())}
            )
        if input_data["Current_Mood"] not in current_mood_mapping:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Current_Mood value", "valid_values": list(current_mood_mapping.keys())}
            )
        
        # Additional validations for numerical inputs
        if not (1 <= input_data["Parent_Satisfaction"] <= 5):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Parent_Satisfaction value", "valid_range": "1 to 5"}
            )
        
        if not (1 <= input_data["Engagement_Level"] <= 5):
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Engagement_Level value", "valid_range": "1 to 5"}
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
        return {"prediction": pred_label, "suggestions": suggestions}
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
        if not file.content_type.startswith('video/'):
            raise HTTPException(400, "File must be a video")

        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        # Process video frames
        cap = cv2.VideoCapture(temp_video_path)
        emotions = []
        frame_count = 0
        max_frames = 100  # Limit frames to process for performance

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame (e.g., every 5th frame)
            if frame_count % 5 == 0:
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
                    "frame": frame_count,
                    "emotion": predicted_emotion,
                    "confidence": confidence
                })

            frame_count += 1

        cap.release()
        
        # Calculate emotion statistics
        emotion_stats = {}
        for e in emotion_labels:
            count = sum(1 for x in emotions if x["emotion"] == e)
            emotion_stats[e] = {
                "count": count,
                "percentage": (count / len(emotions)) * 100 if emotions else 0
            }

        # Get dominant emotion (emotion with highest percentage)
        dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1]["percentage"])[0] if emotions else None

        return {
            "frame_analysis": emotions,
            "statistics": emotion_stats,
            "dominant_emotion": dominant_emotion,
            "total_frames_processed": len(emotions)
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

import cv2
import numpy as np
import torch
import json
from fastapi import FastAPI
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine, euclidean
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Real-Time Face Recognition API", version="2.0")

# Load InsightFace model (for face detection & embedding extraction)
face_analysis = FaceAnalysis(name="buffalo_l")
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

# Face embeddings storage
DB_FILE = "face_db.json"

# Load existing face embeddings
try:
    with open(DB_FILE, "r") as f:
        known_faces = json.load(f)
except FileNotFoundError:
    known_faces = {}

# Save updated face database
def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(known_faces, f, indent=4)

# Face matching function
def is_match(embedding1, embedding2, cosine_threshold=0.6, euclidean_threshold=1.0):
    cos_sim = 1 - cosine(embedding1, embedding2)
    euc_dist = euclidean(embedding1, embedding2)
    return cos_sim > cosine_threshold and euc_dist < euclidean_threshold

# API Model for face registration
class FaceRegister(BaseModel):
    name: str

# Route: Register a New Face (Using Camera)
@app.post("/register-face-camera/")
def register_face_camera(data: FaceRegister):
    name = data.name

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Failed to open camera"}

    while True:
        ret, frame = cap.read()
        if not ret:
            return {"error": "Failed to capture image"}

        # Convert frame to RGB (InsightFace requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_analysis.get(rgb_frame)

        if len(faces) > 0:
            # Extract first detected face
            face_embedding = faces[0].normed_embedding.tolist()

            # Store face in database
            known_faces[name] = face_embedding
            save_db()

            cap.release()
            cv2.destroyAllWindows()
            return {"message": "Face registered successfully", "name": name}

        # Display frame with instructions
        cv2.putText(frame, "Face not detected. Please align properly!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Register Face", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"error": "Face registration failed"}

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

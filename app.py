import os

import cv2
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from capture import capture_face
import mediapipe as mp  # ✅ Added MediaPipe for better face detection
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from models import Employee, CheckIn, SessionLocal
from pydantic import BaseModel
from recognizer import get_embeddings, is_match
from capture import enhance_face_if_needed
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, DELETE, etc.
    allow_headers=["*"],  # Allows all headers
)

EMPLOYEE_FOLDER = "employee_faces/"
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class EmployeeCreate(BaseModel):
    employee_id: str
    name: str

class CheckInRequest(BaseModel):
    employee_id: str

@app.get("/check/")
def check_status():
    return "chal raha hai"

@app.post("/add_employee/")
def register_employee(employee: EmployeeCreate, db: Session = Depends(get_db)):
    image_path = capture_face(employee.employee_id)

    if not image_path:
        raise HTTPException(status_code=400, detail="Face capture failed.")

    print(f"Image saved at: {image_path}")

    arcface_emb, vit_emb = get_embeddings(image_path)

    if arcface_emb is None or vit_emb is None:
        raise HTTPException(status_code=500, detail="Error generating embeddings!")

    new_employee = Employee(
        employee_id=employee.employee_id,
        name=employee.name,
        image_path=image_path,
        arcface_embedding=arcface_emb.tolist(),
        vit_embedding=vit_emb.tolist()
    )

    db.add(new_employee)
    db.commit()
    
    return {"message": "Employee registered successfully!"}

@app.delete("/delete-employee/{employee_id}")
def delete_employee(employee_id: str, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found.")

    if os.path.exists(employee.image_path):
        os.remove(employee.image_path)

    db.delete(employee)
    db.commit()

    return {"message": f"✅ Employee {employee_id} deleted successfully!"}

@app.post("/checkin/")
def checkin_employee(db: Session = Depends(get_db)):
    image_path = capture_face("temp_checkin")
    
    if not image_path:
        raise HTTPException(status_code=400, detail="Face capture failed.")

    enhanced_path = enhance_face_if_needed(image_path)

    if not os.path.exists(enhanced_path):
        raise HTTPException(status_code=500, detail="Enhanced image not found!")

    arc_emb, vit_emb = get_embeddings(enhanced_path)

    if arc_emb is None or vit_emb is None:
        raise HTTPException(status_code=500, detail="Error generating embeddings!")

    for emp in db.query(Employee).all():
        if is_match(arc_emb, np.array(emp.arcface_embedding), vit_emb, np.array(emp.vit_embedding)):
            checkin_entry = CheckIn(employee_id=emp.employee_id)
            db.add(checkin_entry)
            db.commit()
            return {"message": f"✅ {emp.name} checked in!", "time": checkin_entry.checkin_time}

    raise HTTPException(status_code=401, detail="Unauthorized check-in.")

@app.get("/checkins/")
def get_checkins(db: Session = Depends(get_db)):
    checkins = db.query(CheckIn).all()
    return [{"employee_id": c.employee_id, "time": c.checkin_time} for c in checkins]

# ✅ Video Streaming Endpoint
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

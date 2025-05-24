import cv2
import numpy as np
import requests
import os
from deepface import DeepFace

# FastAPI Check-in Endpoint
CHECKIN_URL = "http://127.0.0.1:8000/checkin/"

# Capture Face from Webcam
def capture_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1)
            if key == ord("c"):  # Press 'c' to capture
                temp_image_path = "temp_face.jpg"
                cv2.imwrite(temp_image_path, face_img)
                cap.release()
                cv2.destroyAllWindows()
                return temp_image_path  # ✅ Return file path instead

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


# Extract Embedding & Recognize Employee
def recognize_and_checkin():
    face_img = capture_face()
    
    if face_img is None:
        print("No face captured.")
        return

    try:
        # Save temporary face image
        temp_image_path = "temp_face.jpg"
        cv2.imwrite(temp_image_path, face_img)

        # Extract embedding using DeepFace
        embedding = DeepFace.represent(img_path=temp_image_path, model_name="Facenet")[0]["embedding"]
        
        # Send embedding to FastAPI check-in endpoint
        response = requests.post(CHECKIN_URL, json={"embedding": embedding})
        
        if response.status_code == 200:
            print("✅ Check-in Successful:", response.json())
        else:
            print("❌ Check-in Failed:", response.json())

        os.remove(temp_image_path)  # Delete temp image after use

    except Exception as e:
        print(f"Error: {str(e)}")

# Run Face Recognition Check-in
if __name__ == "__main__":
    recognize_and_checkin()

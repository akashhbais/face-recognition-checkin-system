import cv2
import os
import torch
import threading
import numpy as np
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis

# Set up directories
CAPTURED_FACE_DIR = "employee_faces/"
os.makedirs(CAPTURED_FACE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load RetinaFace with GPU support
try:
    retinaface = FaceAnalysis(name="buffalo_l")
    retinaface.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    print("RetinaFace Loaded Successfully!")
except Exception as e:
    print(f"RetinaFace Failed to Load: {e}")
    retinaface = None  

# ✅ Load GFPGAN for face enhancement (Lowered Intensity)
gfpgan = GFPGANer(model_path="GFPGANv1.3.pth", upscale=1, device=device)

def is_good_image(image):
    """Checks if the image is already sharp and clear to avoid unnecessary enhancement."""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    brightness = np.mean(gray)  
    contrast = gray.std()       

    BLUR_THRESHOLD = 100.0
    BRIGHTNESS_MIN = 60.0        
    BRIGHTNESS_MAX = 200.0       
    CONTRAST_THRESHOLD = 20.0    

    if laplacian_var > BLUR_THRESHOLD and BRIGHTNESS_MIN < brightness < BRIGHTNESS_MAX and contrast > CONTRAST_THRESHOLD:
        return True  
    return False  

def enhance_face_if_needed(image_path):
    """Enhances the captured face only if needed and ensures it looks natural."""
    
    img = cv2.imread(image_path)
    
    if img is None or img.size == 0:
        print(f"Error: Could not load image: {image_path}")
        return image_path  

    if is_good_image(img):  
        print("Image is already good, skipping enhancement.")
        return image_path  

    print("Enhancing image for better recognition...")

    try:
        # Using GFPGAN to enhance the face
        result = gfpgan.enhance(img, has_aligned=False, only_center_face=True, paste_back=False)

        if not isinstance(result, tuple) or len(result) < 2:
            print("GFPGAN returned an invalid result. Keeping original.")
            return image_path
        
        restored_face = result[1]  

        # Fix: Ensure restored_face is an np.ndarray and is not empty
        if isinstance(restored_face, list):  
            print(f"Converting restored face from list to np.ndarray. List length: {len(restored_face)}")
            restored_face = np.array(restored_face, dtype=np.uint8)

        if not isinstance(restored_face, np.ndarray) or restored_face.size == 0:
            print(f"Unexpected or empty restored face. Type: {type(restored_face)}, Size: {restored_face.size if isinstance(restored_face, np.ndarray) else 'N/A'}")
            return image_path

        # Ensure dtype is uint8 for OpenCV compatibility
        if restored_face.dtype != np.uint8:
            print("Converting restored face to uint8.")
            restored_face = restored_face.astype(np.uint8)

        # Fix: Ensure restored_face has valid dimensions before resizing
        if restored_face.ndim != 3 or restored_face.shape[0] == 0 or restored_face.shape[1] == 0:
            print(f"Invalid restored face dimensions: {restored_face.shape}")
            return image_path

        # Resize if dimensions mismatch
        if img.shape[:2] != restored_face.shape[:2]:
            print(f"Resizing restored face from {restored_face.shape[:2]} to {img.shape[:2]}")
            restored_face = cv2.resize(restored_face, (img.shape[1], img.shape[0]))

        # Blending with original image (soft enhancement)
        blend_ratio = 0.15  
        blended_face = cv2.addWeighted(img, 1 - blend_ratio, restored_face, blend_ratio, 0)

        enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")
        cv2.imwrite(enhanced_path, blended_face)
        print(f"Enhanced image saved at: {enhanced_path}")
        return enhanced_path  

    except Exception as e:
        print(f"GFPGAN Enhancement Error: {e}")
        return image_path



def capture_face(employee_id):
    """Captures face using webcam and processes it."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return None

    face_path = os.path.join(CAPTURED_FACE_DIR, f"{employee_id}.jpg")
    max_attempts = 5  
    attempt = 0

    print("📸 Press 'C' to capture face | 'Q' to quit.")

    def process_face(frame):
        """Runs face detection & enhancement asynchronously."""
        if retinaface is None:
            print("RetinaFace not initialized. Cannot detect faces.")
            return False

        faces = retinaface.get(frame)
        if not faces:
            print("No face detected. Adjust lighting & try again.")
            return False

        faces = sorted(faces, key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)
        bbox = faces[0].bbox.astype(int)

        PAD_RATIO_W = 0.25  
        PAD_RATIO_H = 0.2   

        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(PAD_RATIO_W * width), int(PAD_RATIO_H * height)

        x = max(0, bbox[0] - pad_x)
        y = max(0, bbox[1] - pad_y)
        x2 = min(frame.shape[1], bbox[2] + pad_x)
        y2 = min(frame.shape[0], bbox[3] + pad_y)

        face_img = frame[y:y2, x:x2]
        if face_img.size == 0:
            print("Extracted face image is empty. Skipping enhancement.")
            return False

        cv2.imwrite(face_path, face_img)
        print(f"Image saved at: {face_path}")

        enhanced_path = enhance_face_if_needed(face_path)
        return enhanced_path is not None

    while attempt < max_attempts:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam. Retrying...")
            attempt += 1
            continue

        cv2.putText(frame, "Press 'C' to Capture | 'Q' to Quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(50) & 0xFF  

        if key == ord("c"):
            print("'C' Key Detected - Capturing Image...")
            if process_face(frame):
                print(f"Face capture successful: {face_path}")
                break  
            else:
                attempt += 1
                print(f"Retrying face capture ({attempt}/{max_attempts})...")

        elif key == ord("q"):
            print("Quit signal received. Closing camera.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return face_path if attempt < max_attempts else None

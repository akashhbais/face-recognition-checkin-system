import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from scipy.spatial.distance import cosine

# ✅ Load RetinaFace
try:
    face_detector = FaceAnalysis(name="buffalo_l")
    face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    print("✅ RetinaFace Loaded Successfully!")
except Exception as e:
    print(f"❌ RetinaFace Failed to Load: {e}")
    face_detector = None

# ✅ Load Vision Transformer (ViT)
vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
vit_model.eval()

vit_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embeddings(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return None, None

    # ✅ Get Face Embedding from RetinaFace
    faces = face_detector.get(img) if face_detector else []
    if not faces:
        print("❌ No face detected in image.")
        return None, None  # Prevents NoneType errors

    arcface_emb = faces[0].normed_embedding

    # ✅ Convert Image for ViT Model
    img_tensor = vit_transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # No gradients required for inference
        vit_emb = vit_model(img_tensor).numpy().flatten()

    # ✅ Normalize ViT Embedding
    vit_emb = vit_emb / np.linalg.norm(vit_emb)

    return arcface_emb, vit_emb

def is_match(arc_emb1, arc_emb2, vit_emb1, vit_emb2, threshold=0.65):
    """Compares embeddings using cosine similarity."""
    if arc_emb1 is None or arc_emb2 is None or vit_emb1 is None or vit_emb2 is None:
        return False

    arc_sim = 1 - cosine(arc_emb1, arc_emb2)
    vit_sim = 1 - cosine(vit_emb1, vit_emb2)

    final_score = (arc_sim * 0.7) + (vit_sim * 0.3)
    return final_score > threshold

# modules/recognize_face.py

import cv2
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os

embedder = FaceNet()
detector = MTCNN()

def load_user_embedding(name, aadhar):
    path = os.path.join("embeddings", f"{name}_{aadhar}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Embedding file not found: " + path)
    with open(path, "rb") as f:
        emb = pickle.load(f)
    return emb

def variance_of_laplacian(image_gray):
    # measure of focus (sharpness). Low -> blurry / flat
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()

def verify_face_live(name, aadhar, threshold=0.6, max_attempts=3):
    """
    Opens webcam, waits for 's' press to take image and verify.
    Returns: dict { matched: bool, similarity: float or None, reason: str, spoof_score: float }
    """
    saved_emb = load_user_embedding(name, aadhar)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam for verification")

    print("Live verification: press 's' to verify, 'q' to quit.")
    attempts = 0
    result = {"matched": False, "similarity": None, "reason": "No attempt", "spoof_score": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, "Press 's' to verify or 'q' to quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Verification", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            attempts += 1
            faces = detector.detect_faces(frame)

            if len(faces) == 0:
                print("❌ No face detected. Try again.")
                result["reason"] = "no_face"
                if attempts >= max_attempts:
                    break
                continue

            x, y, w, h = faces[0]['box']
            # clamp coords
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]

            # anti-spoof check: blur / flatness
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            lap_var = variance_of_laplacian(gray)
            result["spoof_score"] = float(lap_var)
            # heuristic: very low variance (<80) likely printed/screenshot; tune for your camera
            if lap_var < 80:
                print("⚠️ Low texture / blurry face detected (possible spoof). LapVar:", lap_var)
                result["reason"] = "low_texture_spoof"
                # still continue to compute embedding to log similarity if needed

            # preprocess & embedding
            try:
                img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (160,160))
            except Exception as e:
                print("Error preparing face for embedding:", e)
                result["reason"] = "prep_error"
                if attempts >= max_attempts:
                    break
                continue

            live_emb = embedder.embeddings([img_resized])[0]

            # similarity (cosine)
            sim = cosine_similarity([saved_emb], [live_emb])[0][0]
            result["similarity"] = float(sim)
            print("Cosine similarity:", sim)

            if sim >= threshold and lap_var >= 80:
                print("✔ FACE MATCHED (and passed simple spoof check)")
                result["matched"] = True
                result["reason"] = "matched"
            elif sim >= threshold and lap_var < 80:
                # matched but spoof suspicion
                print("⚠️ FACE matched but low texture suspicious (log as fraud attempt).")
                result["matched"] = True
                result["reason"] = "matched_but_low_texture"
            else:
                print("✖ FACE DID NOT MATCH")
                result["matched"] = False
                result["reason"] = "low_similarity"

            break

        elif key == ord('q'):
            print("Verification cancelled.")
            result["reason"] = "cancelled"
            break

    cap.release()
    cv2.destroyAllWindows()
    return result

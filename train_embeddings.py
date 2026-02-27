# modules/train_embeddings.py
import os
import cv2
import numpy as np
import pickle
import csv
from keras_facenet import FaceNet

EMBEDDINGS_CSV = os.path.join("embeddings", "embeddings.csv")
embedder = FaceNet()

def ensure_embeddings_csv(header_len=None):
    if not os.path.exists(EMBEDDINGS_CSV):
        # header will be: aadhar,img_name,e1,e2,...eN (created dynamically on first write)
        with open(EMBEDDINGS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["aadhar","img_name","e0"])  # placeholder; will be overwritten on first append if necessary

def image_to_embedding(img):
    # expects BGR image from cv2
    # FaceNet from keras-facenet accepts RGB images; resizing done here
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160,160))
    embedding = embedder.embeddings([img_resized])[0]
    return embedding

def append_embeddings_to_csv(aadhar, img_name, embedding):
    # write header dynamically once using length of embedding
    emb_len = len(embedding)
    if not os.path.exists(EMBEDDINGS_CSV):
        header = ["aadhar","img_name"] + [f"e{i}" for i in range(emb_len)]
        with open(EMBEDDINGS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(EMBEDDINGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([aadhar, img_name] + list(map(float, embedding)))

def generate_embeddings_for_user(name, aadhar):
    folder = os.path.join("dataset", f"{name}_{aadhar}")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"User folder not found: {folder}")

    embeddings = []
    print(f"\nProcessing folder: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))])

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print("WARNING: cannot read", path)
            continue

        # show small pixel snippet
        print(f"\n--- Image {i+1}: {path} ---")
        print("Pixel snippet (first 3x3 RGB pixels):")
        snippet = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:3, :3, :]
        print(snippet.tolist())

        emb = image_to_embedding(img)
        print("Embedding snippet (first 10 values):")
        print(np.round(emb[:10], 6).tolist())

        embeddings.append(emb)
        append_embeddings_to_csv(aadhar, fname, emb)

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings generated for this user.")

    # Final embedding = mean of per-image embeddings
    final_emb = np.mean(np.vstack(embeddings), axis=0)
    os.makedirs("embeddings", exist_ok=True)
    save_path = os.path.join("embeddings", f"{name}_{aadhar}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(final_emb, f)

    print("\n=== FINAL USER EMBEDDING ===")
    print(np.round(final_emb, 6).tolist())
    print("Saved final embedding to:", save_path)
    return save_path

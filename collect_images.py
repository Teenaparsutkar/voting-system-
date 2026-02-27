# modules/collect_images.py
import cv2
import os
import time
from mtcnn import MTCNN

def capture_images(name, aadhar, num_images=100):
    folder_name = f"{name}_{aadhar}"
    user_folder = os.path.join("dataset", folder_name)
    os.makedirs(user_folder, exist_ok=True)

    detector = MTCNN()  # Face Detector

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera or index.")

    print("Auto-capturing FACE images. Look at the camera…")

    count = 0
    last_capture_time = time.time()

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        faces = detector.detect_faces(frame)

        if faces:
            x, y, w, h = faces[0]['box']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            face_crop = frame[y:y+h, x:x+w]

            # Capture image every 0.6 sec
            if time.time() - last_capture_time > 0.6:
                img_path = os.path.join(user_folder, f"img_{count+1}.jpg")
                cv2.imwrite(img_path, face_crop)
                print(f"Auto-saved: {img_path}")
                count += 1
                last_capture_time = time.time()
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Image {count}/{num_images}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Auto Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return user_folder

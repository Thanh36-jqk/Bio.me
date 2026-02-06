import os, time, json, glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ===== Config (folders must already exist) =====
DATASET_DIR = Path("data_faces")          # dataset/<person>/*.jpg|png
ARTIFACTS_DIR = Path("models")            # contains lbph_model.yml and labels.json
MODEL_PATH = ARTIFACTS_DIR / "lbph_model.yml"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"

# LBPH & Preprocess
LBPH_RADIUS = 2
LBPH_NEIGHBORS = 16
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
FACE_SIZE = (100, 100)
MIN_FACE = 64


# Evaluation
RANDOM_SEED = 42
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ===== Basic checks =====
if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    raise RuntimeError("cv2.face.LBPHFaceRecognizer_create not available. Install opencv-contrib-python.")

# NOTE: Folder existence checks disabled to allow importing from different working directories
# These checks will run when the actual functions (train/load) are called
# if not DATASET_DIR.exists():
#     raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")
# if not ARTIFACTS_DIR.exists():
#     raise FileNotFoundError(f"Artifacts folder not found: {ARTIFACTS_DIR.resolve()}")

# ===== Face utils =====
CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

def detect_faces_bgr(image_bgr, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    return faces, gray

def preprocess_crop(gray, x, y, w, h, out_size=(200, 200)):
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
    roi = cv2.equalizeHist(roi)
    return roi

# ===== Data loading =====
def collect_images_and_labels(dataset_dir: Path) -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    label2id: Dict[str, int] = {}
    next_id = 0
    total_files = 0
    used_files = 0

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        if label not in label2id:
            label2id[label] = next_id
            next_id += 1

        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                continue
            total_files += 1
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # If looks cropped, skip detection; else detect and crop the largest face.
            h, w = gray.shape
            is_probably_cropped = (min(h, w) / max(h, w) > 0.85)
            faces = []
            if not is_probably_cropped:
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(MIN_FACE, MIN_FACE))

            if len(faces) == 0:
                face_roi = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
                images.append(face_roi)
                labels.append(label2id[label])
                used_files += 1
            else:
                x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
                if min(w, h) < MIN_FACE:
                    continue
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
                images.append(roi)
                labels.append(label2id[label])
                used_files += 1

    if len(images) == 0:
        raise RuntimeError(f"No valid face images to train from '{dataset_dir}'. Expected 'dataset/<label>/*.jpg'.")

    print(f"[INFO] Read files: {total_files}, used for train: {used_files}, classes: {len(label2id)}")
    return images, labels, label2id

# ===== Train & Save =====
def save_lbph_safely(recognizer, model_path: Path):
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    recognizer.write(str(tmp_path))
    os.replace(tmp_path, model_path)

def train_lbph(dataset_dir: Path, model_path: Path, labels_path: Path):
    images, labels, label2id = collect_images_and_labels(dataset_dir)
    labels_np = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    print("[INFO] Training LBPH...")
    recognizer.train(images, labels_np)

    print(f"[INFO] Saving model → {model_path}")
    save_lbph_safely(recognizer, model_path)

    print(f"[INFO] Saving labels → {labels_path}")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # quick read-back sanity check
    test_rec = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    test_rec.read(str(model_path))
    print("[OK] Model read-back success.")

# ===== Load model =====
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Model or labels not found. Train first.")

    if MODEL_PATH.stat().st_size < 100:
        raise ValueError(f"LBPH model file looks corrupted/empty ({MODEL_PATH}, size={MODEL_PATH.stat().st_size}B).")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    recognizer.read(str(MODEL_PATH))
    return recognizer, id2label

# ===== Utilities =====
def compute_confidence(dist, dist_threshold=130):
    if dist is None or np.isnan(dist):
        return 0.0
    conf = (float(dist_threshold) - float(dist)) / float(dist_threshold)
    return float(max(0.0, min(1.0, conf)) * 100.0)

def evaluate_with_sklearn(images: List[np.ndarray], labels: List[int], id2label: Dict[int, str]):
    if not SKLEARN_OK or len(images) < 4 or len(set(labels)) < 2:
        print("[NOTE] Skip evaluation (missing sklearn or insufficient data).")
        return

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    rec.train([img for img in X_train], y_train)
    preds = []
    for img in X_test:
        label_id, dist = rec.predict(img)
        preds.append(label_id)

    target_names = [id2label[i] for i in sorted(set(y_test))]
    print(classification_report(y_test, preds, target_names=target_names))
    print("Accuracy:", accuracy_score(y_test, preds))

# ===== Realtime recognition =====
def realtime_recognition(cam_index=0, dist_threshold=130, debug=False):
    recognizer, id2label = load_model()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            faces, gray = detect_faces_bgr(frame, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
            for (x, y, w, h) in faces:
                face_img = preprocess_crop(gray, x, y, w, h, out_size=(200, 200))
                label_id, dist = recognizer.predict(face_img)
                conf = compute_confidence(dist, dist_threshold)
                if debug:
                    print(f"[DEBUG] id={label_id}, name={id2label.get(label_id,'?')}, dist={dist:.2f}, conf={conf:.1f}")
                if dist <= dist_threshold:
                    name = id2label.get(label_id, "UNKNOWN"); color = (0, 255, 0)
                else:
                    name = "UNKNOWN"; color = (0, 0, 255)
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} | conf:{conf:.1f} | dist:{dist:.1f}", (x, max(0,y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("Real-time Recognition (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ===== Main =====
def main():
    # Train only if no model
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("[INFO] No model found. Training...")
        train_lbph(DATASET_DIR, MODEL_PATH, LABELS_PATH)
    else:
        try:
            _ = load_model()
            print("[OK] Existing model is readable. Skip training.")
        except Exception as e:
            print(f"[WARN] Model not readable ({e}). Retraining...")
            train_lbph(DATASET_DIR, MODEL_PATH, LABELS_PATH)

    # Optional evaluation
    try:
        images, labels, label2id = collect_images_and_labels(DATASET_DIR)
        id2label = {v: k for k, v in label2id.items()}
        evaluate_with_sklearn(images, labels, id2label)
    except Exception as e:
        print(f"[NOTE] Skip evaluation: {e}")

    # Start realtime
    realtime_recognition(cam_index=0, dist_threshold=130, debug=True)

if __name__ == "__main__":
    main()

"""
Biometric Multi-Factor Authentication System
Sequential authentication: Name Input -> Face -> Iris -> Fingerprint -> Welcome
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
import streamlit as st

# ========== Configuration ==========
ROOT = Path(__file__).resolve().parent.parent  # Go up from apps/ to MIDTERM/
FACIAL_DIR = ROOT / "modules" / "facial"
IRIS_DIR = ROOT / "modules" / "iris"
FP_DIR = ROOT / "modules" / "fingerprint"

for path in [FACIAL_DIR, IRIS_DIR, FP_DIR]:
    sys.path.insert(0, str(path))

import Facial as facial_mod
import iris as iris_mod
import fingerprint as fp_mod

st.set_page_config(page_title="Biometric MFA", layout="wide", initial_sidebar_state="collapsed")

# ========== Helper Functions ==========
def init_session():
    """Initialize session state variables"""
    defaults = {"auth_step": 0, "user_name": "", "face_ok": False, "face_name": None, "face_dist": None,
                "iris_ok": False, "iris_name": None, "iris_dist": None, "fp_ok": False, "fp_name": None, "fp_count": None}
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

def to_bgr(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert image to BGR format"""
    if image is None or len(image.shape) == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR if len(image.shape) == 2 else cv2.COLOR_BGRA2BGR)

def clean_name_for_match(name: str) -> str:
    """Clean biometric database name for matching (remove extensions, suffixes, numbers)"""
    import re
    import unicodedata
    
    if not name:
        return ""
    
    # Remove file extension
    name = name.split('.')[0]
    
    # Remove common suffixes like " right (2)", " left", " (1)", etc.
    name = re.sub(r'\s+(right|left|center)\s*\(\d+\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+\(\d+\)', '', name)
    name = re.sub(r'\s+(right|left|center)$', '', name, flags=re.IGNORECASE)
    
    # Normalize Unicode (NFC normalization for Vietnamese)
    name = unicodedata.normalize('NFC', name)
    
    return name.strip()

def remove_accents(text: str) -> str:
    """Remove Vietnamese accents for fuzzy matching"""
    import unicodedata
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

def names_match(name1: str, name2: str, strict: bool = True) -> bool:
    """
    Compare two names with optional fuzzy matching.
    If strict=False, compares without accents for Vietnamese names.
    """
    if not name1 or not name2:
        return False
    
    if name1.lower() == name2.lower():
        return True
    
    if not strict:
        return remove_accents(name1).lower() == remove_accents(name2).lower()
    
    return False

def show_progress():
    """Display progress indicator"""
    steps = ["Nhap ten", "Face", "Iris", "Fingerprint", "Hoan tat"]
    checks = [bool(st.session_state.user_name), st.session_state.face_ok, st.session_state.iris_ok, 
              st.session_state.fp_ok, st.session_state.auth_step == 4]
    
    cols = st.columns(5)
    for i, (col, step, done) in enumerate(zip(cols, steps, checks)):
        with col:
            icon = "[DONE]" if done else "[NOW]" if st.session_state.auth_step == i else "[ ]"
            col.markdown(f"### {icon} {step}")

def display_auth_result(ok: bool, name: str, metric: str, threshold: str):
    """Display authentication result with status"""
    status = "PASS" if ok else "FAIL"
    color = "green" if ok else "red"
    st.markdown(f"### Ket qua: :{color}[{status}]")
    st.write(f"**Nhan dien:** {name} | **{metric}:** {threshold}")
    if ok:
        st.success(f"Xac thuc thanh cong!")
    else:
        st.error(f"Xac thuc that bai")

# ========== Load Models ==========
@st.cache_resource(show_spinner=True)
def load_models():
    """Load all biometric models"""
    cwd = os.getcwd()
    os.chdir(str(FACIAL_DIR))
    try:
        recog, id2label = facial_mod.load_model()
    finally:
        os.chdir(cwd)
    
    # Build iris database
    iris_db = {}
    base = IRIS_DIR / "mong mat" / ("iris" if (IRIS_DIR / "mong mat" / "iris").exists() else "")
    for p in sorted((base or IRIS_DIR / "mong mat").glob("*")):
        if p.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg"]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                try:
                    img_clean = iris_mod.remove_specular_reflections(img)
                    pupil, iris = iris_mod.detect_iris_boundaries(img_clean)
                    if pupil and iris:
                        polar = iris_mod.normalize_iris(img_clean, pupil, iris, radials=64, angles=512)
                        iris_db[p.name] = {"code": iris_mod.gabor_encode(polar), "pupil": pupil, "iris": iris}
                except:
                    pass
    
    # Build fingerprint database
    fp_db = fp_mod.load_dataset_descriptors(str(FP_DIR / "van tay"))
    
    return recog, id2label, iris_db, fp_db

recog, id2label, iris_db, fp_db = load_models()

# ========== Authentication Functions ==========
def check_face(img_bgr: np.ndarray, threshold: int = 130):
    """Face recognition check"""
    faces, gray = facial_mod.detect_faces_bgr(img_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return False, None, None, img_bgr
    
    best = min([(x, y, w, h, *recog.predict(facial_mod.preprocess_crop(gray, x, y, w, h, (200, 200)))) 
                for x, y, w, h in faces], key=lambda b: b[5], default=None)
    
    if not best:
        return False, None, None, img_bgr
    
    x, y, w, h, lid, dist = best
    ok, name = dist <= threshold, id2label.get(lid, f"id_{lid}")
    dbg = img_bgr.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(dbg, f"{name} d={dist:.1f}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return ok, name, dist, dbg

def check_iris(img_bgr: np.ndarray, max_dist: float = 0.35):
    """Iris verification check"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_clean = iris_mod.remove_specular_reflections(gray)
    pupil, iris = iris_mod.detect_iris_boundaries(img_clean)
    
    if not pupil or not iris:
        return False, None, 1.0, to_bgr(img_bgr)
    
    polar = iris_mod.normalize_iris(img_clean, pupil, iris, radials=64, angles=512)
    code = iris_mod.gabor_encode(polar)
    
    best_name, best_dist = None, 1.0
    for name, rec in iris_db.items():
        dist, _ = iris_mod.hamming_distance(code, rec["code"])
        if dist < best_dist:
            best_dist, best_name = dist, name
    
    ok = best_dist <= max_dist
    dbg = to_bgr(img_bgr.copy())
    if pupil and iris:
        cv2.circle(dbg, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (0, 255, 0) if ok else (0, 0, 255), 2)
        cv2.circle(dbg, (int(iris[0]), int(iris[1])), int(iris[2]), (0, 255, 0) if ok else (0, 0, 255), 2)
    return ok, best_name, best_dist, dbg

def check_fingerprint(img_bgr: np.ndarray, min_match: int = 15):
    """Fingerprint matching check"""
    orb = cv2.ORB_create(nfeatures=1500)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    proc = fp_mod.process_image(gray)
    kp1, des1 = orb.detectAndCompute(proc, None)
    
    if des1 is None:
        return False, None, 0
    
    best_id, best_count = None, -1
    for person, kp_train, des_train, _ in fp_db:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des_train)
        good = [m for m in matches if m.distance < 60]
        if len(good) > best_count:
            best_count, best_id = len(good), person
    
    return best_count >= min_match, best_id, best_count

# ========== UI ==========
st.title("Biometric Multifactor Authentication")
st.caption("Yeu cau: **Ten + Mat + Iris + Van tay** cung khop")

init_session()
st.markdown("---")
show_progress()
st.markdown("---")

# Step 0: Name Input
if st.session_state.auth_step == 0:
    st.header("Nhap ten cua ban")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        name = st.text_input("Ten cua ban:", st.session_state.user_name, placeholder="Vi du: Nguyen Van A")
        if name:
            st.session_state.user_name = name
            st.success(f"Xin chao, {name}!")
            if st.button(">> Bat dau xac thuc", type="primary"):
                st.session_state.auth_step = 1
                st.rerun()
        else:
            st.info("Nhap ten de tiep tuc")
    
    with col2:
        st.markdown("### Huong dan:\n1. Nhap ten day du\n2. Xac thuc qua 3 buoc\n3. Hoan tat dang nhap")

# Step 1: Face
elif st.session_state.auth_step == 1:
    st.header("Buoc 1: Xac thuc khuon mat")
    thresh = st.slider("Nguong LBPH", 60, 200, 130, 5)
    
    col1, col2 = st.columns(2)
    with col1:
        cam = st.camera_input("Chup anh")
        if cam:
            img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            ok, name, dist, dbg = check_face(img, thresh)
            st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
            
            # Check if detected name matches entered name
            face_name_clean = clean_name_for_match(name) if name else None
            user_name_clean = clean_name_for_match(st.session_state.user_name)
            
            # Try exact match first, then fuzzy match (without accents)
            name_match = names_match(face_name_clean, user_name_clean, strict=False)
            final_ok = ok and name_match
            
            st.session_state.face_ok = final_ok
            st.session_state.face_name = face_name_clean
            st.session_state.face_dist = dist
            
            display_auth_result(final_ok, face_name_clean or "N/A", "Khoang cach", f"{dist:.1f}")
            
            if ok and not name_match:
                st.warning(f"Khong khop: '{face_name_clean}' != '{user_name_clean}'")
                st.info(f"Goi y: Kiem tra dau tieng Viet. Khong dau: '{remove_accents(face_name_clean)}' vs '{remove_accents(user_name_clean)}'")
    
    with col2:
        st.markdown(f"### Huong dan:\n1. Chup anh ro mat\n2. Ten nhan dien phai khop: **{st.session_state.user_name}**\n3. Nhan Tiep tuc neu PASS")
        
        if st.session_state.face_ok:
            st.success(f"[OK] {st.session_state.face_name}")
            if st.button(">> Tiep tuc -> Iris", type="primary"):
                st.session_state.auth_step = 2
                st.rerun()
        else:
            st.error("Chua xac thuc hoac ten khong khop")

# Step 2: Iris
elif st.session_state.auth_step == 2:
    st.header("Buoc 2: Xac thuc Iris")
    max_d = st.slider("Nguong Hamming", 0.1, 0.6, 0.35, 0.01)
    
    col1, col2 = st.columns(2)
    with col1:
        iris_file = st.file_uploader("Tai anh mat", type=["bmp", "png", "jpg"], key="iris")
        if iris_file:
            img = cv2.imdecode(np.frombuffer(iris_file.read(), np.uint8), cv2.IMREAD_COLOR)
            ok, name, dist, dbg = check_iris(img, max_d)
            st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
            
            # Check name match (clean database names)
            iris_name_clean = clean_name_for_match(name)
            user_name_clean = clean_name_for_match(st.session_state.user_name)
            name_match = names_match(iris_name_clean, user_name_clean, strict=False)
            final_ok = ok and name_match
            
            st.session_state.iris_ok = final_ok
            st.session_state.iris_name = iris_name_clean
            st.session_state.iris_dist = dist
            
            display_auth_result(final_ok, iris_name_clean or "N/A", "Hamming", f"{dist:.3f}")
            
            if ok and not name_match:
                st.warning(f"Canh bao bao mat: Nhan dien '{iris_name_clean}' != '{user_name_clean}'. Khong khop!")
    
    with col2:
        st.markdown(f"### Trang thai:\n- [OK] Face: {st.session_state.face_name}\n- Ten can khop: **{st.session_state.user_name}**")
        
        if st.session_state.iris_ok:
            st.success(f"[OK] Iris: {st.session_state.iris_name}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("<< Quay lai"):
                st.session_state.auth_step = 1
                st.rerun()
        with col_b:
            if st.session_state.iris_ok and st.button(">> Tiep tuc", type="primary"):
                st.session_state.auth_step = 3
                st.rerun()

# Step 3: Fingerprint
elif st.session_state.auth_step == 3:
    st.header("Buoc 3: Xac thuc van tay")
    min_m = st.slider("ORB matches toi thieu", 5, 50, 15)
    
    col1, col2 = st.columns(2)
    with col1:
        fp_file = st.file_uploader("Tai anh van tay", type=["tif", "png", "jpg"], key="fp")
        if fp_file:
            img = cv2.imdecode(np.frombuffer(fp_file.read(), np.uint8), cv2.IMREAD_COLOR)
            ok, name, count = check_fingerprint(img, min_m)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Check name match
            fp_name_clean = clean_name_for_match(name) if name else None
            user_name_clean = clean_name_for_match(st.session_state.user_name)
            name_match = names_match(fp_name_clean, user_name_clean, strict=False)
            final_ok = ok and name_match
            
            st.session_state.fp_ok = final_ok
            st.session_state.fp_name = name
            st.session_state.fp_count = count
            
            display_auth_result(final_ok, name or "N/A", "Matches", str(count))
            
            if ok and not name_match:
                st.warning(f"Canh bao bao mat: Nhan dien duoc '{name}' nhung ban nhap ten '{st.session_state.user_name}'. Khong khop!")
    
    with col2:
        st.markdown(f"### Trang thai:\n- [OK] Face: {st.session_state.face_name}\n- [OK] Iris: {st.session_state.iris_name}\n- Ten can khop: **{st.session_state.user_name}**")
        
        if st.session_state.fp_ok:
            st.success(f"[OK] Fingerprint: {st.session_state.fp_name}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("<< Quay lai"):
                st.session_state.auth_step = 2
                st.rerun()
        with col_b:
            if st.session_state.fp_ok and st.button("[X] Hoan tat", type="primary"):
                st.session_state.auth_step = 4
                st.rerun()

# Step 4: Welcome
elif st.session_state.auth_step == 4:
    st.balloons()
    st.markdown(f"# Xin chao, **{st.session_state.user_name}**!")
    st.markdown("### Xac thuc da yeu to thanh cong!")
    
    st.markdown("---")
    cols = st.columns(3)
    data = [
        ("Face", st.session_state.face_name, f"{st.session_state.face_dist:.1f}"),
        ("Iris", st.session_state.iris_name, f"{st.session_state.iris_dist:.3f}"),
        ("Fingerprint", st.session_state.fp_name, str(st.session_state.fp_count))
    ]
    
    for col, (title, name, metric) in zip(cols, data):
        with col:
            st.markdown(f"### {title}")
            st.success("Da xac thuc")
            st.info(f"**Khop:** {name}\n\n**Metric:** {metric}")
    
    st.markdown("---")
    st.success(f"Truy cap cap phep: {st.session_state.user_name}")
    
    if st.button("Dang nhap nguoi khac", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown("---")
st.caption("Biometric MFA System")

"""
Biometric User Registration System
Allows new users to enroll by uploading 10-15 images per biometric method
"""

import sys
import os
from pathlib import Path
from typing import List

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

st.set_page_config(page_title="Dang ky - Biometric MFA", layout="wide")

# ========== Helper Functions ==========
def init_session():
    """Initialize session state"""
    defaults = {
        "reg_step": 0,  # 0=Name, 1=Face, 2=Iris, 3=Fingerprint, 4=Success
        "username": "",
        "face_images": [],
        "iris_images": [],
        "fp_images": []
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

def save_face_images(username: str, images: List[np.ndarray]) -> bool:
    """Save face images and retrain model"""
    try:
        user_dir = FACIAL_DIR / "data_faces" / username
        user_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images, 1):
            path = user_dir / f"{i}.jpg"
            cv2.imwrite(str(path), img)
        
        # Retrain model
        cwd = os.getcwd()
        os.chdir(str(FACIAL_DIR))
        try:
            facial_mod.train_model()
        finally:
            os.chdir(cwd)
        
        return True
    except Exception as e:
        st.error(f"Loi luu Face: {e}")
        return False

def save_iris_images(username: str, images: List[np.ndarray]) -> bool:
    """Save iris images"""
    try:
        iris_dir = IRIS_DIR / "mong mat"
        iris_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images, 1):
            # Alternate between left and right
            side = "left" if i % 2 == 0 else "right"
            path = iris_dir / f"{username}_{side}_{i}.bmp"
            cv2.imwrite(str(path), img)
        
        return True
    except Exception as e:
        st.error(f"Loi luu Iris: {e}")
        return False

def save_fingerprint_images(username: str, images: List[np.ndarray]) -> bool:
    """Save fingerprint images"""
    try:
        fp_dir = FP_DIR / "van tay"
        fp_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images, 1):
            path = fp_dir / f"{username}_{i}.tif"
            cv2.imwrite(str(path), img)
        
        return True
    except Exception as e:
        st.error(f"Loi luu Fingerprint: {e}")
        return False

def check_username_exists(username: str) -> bool:
    """Check if username already exists in any database"""
    face_exists = (FACIAL_DIR / "data_faces" / username).exists()
    iris_exists = any((IRIS_DIR / "mong mat").glob(f"{username}_*"))
    fp_exists = any((FP_DIR / "van tay").glob(f"{username}_*"))
    return face_exists or iris_exists or fp_exists

# ========== UI ==========
st.title("He thong dang ky sinh trac hoc")
st.caption("Dang ky tai khoan moi voi Face + Iris + Fingerprint")

init_session()

# Progress indicator
st.markdown("---")
steps = ["Nhap ten", "Face (10-15)", "Iris (10-15)", "Fingerprint (10-15)", "Hoan tat"]
cols = st.columns(5)
for i, (col, step) in enumerate(zip(cols, steps)):
    with col:
        icon = "[DONE]" if st.session_state.reg_step > i else "[NOW]" if st.session_state.reg_step == i else "[ ]"
        col.markdown(f"### {icon} {step}")
st.markdown("---")

# Step 0: Enter username
if st.session_state.reg_step == 0:
    st.header("Buoc 1: Nhap ten nguoi dung")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        username = st.text_input("Ten day du:", placeholder="Vi du: Nguyen Van A")
        
        if username:
            if check_username_exists(username):
                st.error(f"Ten '{username}' da ton tai! Hay chon ten khac.")
            else:
                st.success(f"Ten '{username}' kha dung!")
                if st.button(">> Tiep tuc", type="primary"):
                    st.session_state.username = username
                    st.session_state.reg_step = 1
                    st.rerun()
        else:
            st.info("Nhap ten de bat dau")
    
    with col2:
        st.markdown("""
        ### Huong dan:
        1. Nhap ten day du cua ban
        2. Ten khong duoc trung
        3. Sau do se chup/upload 10-15 anh cho moi phuong thuc
        """)

# Step 1: Face images
elif st.session_state.reg_step == 1:
    st.header(f"Buoc 2: Chup anh khuon mat - {st.session_state.username}")
    st.info(f"Da chup: {len(st.session_state.face_images)}/15 anh")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        cam = st.camera_input("Chup anh khuon mat", key=f"cam_{len(st.session_state.face_images)}")
        
        if cam:
            img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            st.session_state.face_images.append(img)
            st.success(f"Da luu anh {len(st.session_state.face_images)}")
            st.rerun()
        
        # Show gallery
        if st.session_state.face_images:
            st.markdown("### Cac anh da chup:")
            cols_gallery = st.columns(5)
            for idx, img in enumerate(st.session_state.face_images):
                with cols_gallery[idx % 5]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Anh {idx+1}", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        ### Huong dan:
        - Can it nhat: **10 anh**
        - Toi da: **15 anh**
        - Chup nhieu goc do khac nhau
        - Anh sang tot
        
        **Tien do:** {len(st.session_state.face_images)}/15
        """)
        
        if len(st.session_state.face_images) >= 10:
            st.success("Du anh! Co the tiep tuc.")
            if st.button(">> Tiep tuc den Iris", type="primary"):
                st.session_state.reg_step = 2
                st.rerun()
        
        if len(st.session_state.face_images) > 0:
            if st.button("Xoa anh cuoi"):
                st.session_state.face_images.pop()
                st.rerun()

# Step 2: Iris images
elif st.session_state.reg_step == 2:
    st.header(f"Buoc 3: Upload anh Iris - {st.session_state.username}")
    st.info(f"Da upload: {len(st.session_state.iris_images)}/15 anh")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "Chon nhieu anh iris (BMP/PNG/JPG)",
            type=["bmp", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="iris_upload"
        )
        
        if uploaded:
            for file in uploaded:
                if len(st.session_state.iris_images) < 15:
                    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        st.session_state.iris_images.append(img)
            st.success(f"Da upload {len(st.session_state.iris_images)} anh")
            st.rerun()
        
        # Gallery
        if st.session_state.iris_images:
            st.markdown("### Cac anh da upload:")
            cols_gallery = st.columns(5)
            for idx, img in enumerate(st.session_state.iris_images):
                with cols_gallery[idx % 5]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Anh {idx+1}", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        ### Huong dan:
        - Can it nhat: **10 anh**
        - Toi da: **15 anh**
        - Upload ca mata trai va phai
        - Anh ro net, khong mo
        
        **Tien do:** {len(st.session_state.iris_images)}/15
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("<< Quay lai"):
                st.session_state.reg_step = 1
                st.rerun()
        
        with col_b:
            if len(st.session_state.iris_images) >= 10:
                if st.button(">> Tiep tuc", type="primary"):
                    st.session_state.reg_step = 3
                    st.rerun()

# Step 3: Fingerprint images
elif st.session_state.reg_step == 3:
    st.header(f"Buoc 4: Upload anh van tay - {st.session_state.username}")
    st.info(f"Da upload: {len(st.session_state.fp_images)}/15 anh")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "Chon nhieu anh van tay (TIF/PNG/JPG)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="fp_upload"
        )
        
        if uploaded:
            for file in uploaded:
                if len(st.session_state.fp_images) < 15:
                    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        st.session_state.fp_images.append(img)
            st.success(f"Da upload {len(st.session_state.fp_images)} anh")
            st.rerun()
        
        # Gallery
        if st.session_state.fp_images:
            st.markdown("### Cac anh da upload:")
            cols_gallery = st.columns(5)
            for idx, img in enumerate(st.session_state.fp_images):
                with cols_gallery[idx % 5]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Anh {idx+1}", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        ### Huong dan:
        - Can it nhat: **10 anh**
        - Toi da: **15 anh**
        - Upload nhieu ngon tay khac nhau
        - Chat luong tot
        
        **Tien do:** {len(st.session_state.fp_images)}/15
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("<< Quay lai"):
                st.session_state.reg_step = 2
                st.rerun()
        
        with col_b:
            if len(st.session_state.fp_images) >= 10:
                if st.button("[X] Hoan tat dang ky", type="primary"):
                    # Save all images
                    with st.spinner("Dang luu du lieu..."):
                        face_ok = save_face_images(st.session_state.username, st.session_state.face_images)
                        iris_ok = save_iris_images(st.session_state.username, st.session_state.iris_images)
                        fp_ok = save_fingerprint_images(st.session_state.username, st.session_state.fp_images)
                        
                        if face_ok and iris_ok and fp_ok:
                            st.session_state.reg_step = 4
                            st.rerun()

# Step 4: Success
elif st.session_state.reg_step == 4:
    st.balloons()
    st.success(f"# Dang ky thanh cong: {st.session_state.username}")
    
    st.markdown("---")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Face")
        st.info(f"Da luu {len(st.session_state.face_images)} anh")
    with cols[1]:
        st.markdown("### Iris")
        st.info(f"Da luu {len(st.session_state.iris_images)} anh")
    with cols[2]:
        st.markdown("### Fingerprint")
        st.info(f"Da luu {len(st.session_state.fp_images)} anh")
    
    st.markdown("---")
    st.info("Ban co the dang nhap bang ung dung streamlit_app.py")
    
    if st.button("Dang ky nguoi khac", type="primary"):
        # Reset
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown("---")
st.caption("Biometric Registration System")

"""
FastAPI Backend for Biometric Multi-Factor Authentication
Supports Face, Iris, and Fingerprint recognition with Deep Learning
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
from pathlib import Path
import logging

# Import recognition modules
from api.face_recognition import FaceRecognizer
from api.iris_recognition import IrisRecognizer
from api.fingerprint_recognition import FingerprintRecognizer
from api.database import db_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Biometric MFA API",
    description="Multi-Factor Authentication using Face, Iris, and Fingerprint",
    version="2.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recognizers
face_recognizer = FaceRecognizer()
iris_recognizer = IrisRecognizer()
fingerprint_recognizer = FingerprintRecognizer()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Connect to MongoDB on startup"""
    await db_manager.connect()
    logger.info("Biometric MFA System started")

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MongoDB on shutdown"""
    await db_manager.disconnect()
    logger.info("Biometric MFA System shutdown")

# Pydantic models
class UserRegistration(BaseModel):
    username: str
    
class AuthenticationResponse(BaseModel):
    success: bool
    username: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    message: str

class RegistrationResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[int] = None

# Helper function
async def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded image file and convert to numpy array"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

# ========== Health Check ==========
@app.get("/")
async def root():
    return {
        "message": "Biometric MFA API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "face": "/api/face/*",
            "iris": "/api/iris/*",
            "fingerprint": "/api/fingerprint/*",
            "user": "/api/user/*"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ========== User Management ==========
@app.post("/api/user/register", response_model=RegistrationResponse)
async def register_user(username: str = Form(...)):
    """Register a new user"""
    try:
        user_id = await db_manager.create_user(username)
        return {"success": True, "user_id": user_id, "message": "User registered successfully"}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/user/{username}")
async def get_user(username: str):
    """Get user information"""
    try:
        user = await db_manager.get_user(username)
        if user:
            return {"exists": True, "user": user}
        return {"exists": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Face Recognition ==========
@app.post("/api/face/register", response_model=RegistrationResponse)
async def register_face(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Register face images for a user (10-15 images recommended)"""
    try:
        if len(files) < 5:
            raise HTTPException(
                status_code=400,
                detail="Minimum 5 face images required"
            )
        
        # Read all images
        images = []
        for file in files:
            img = await read_image_file(file)
            images.append(img)
        
        # Register with face recognizer
        success = face_recognizer.register_user(username, images)
        
        if success:
            # Update database
            await db_manager.update_user_biometric(username, 'face', True)
            return RegistrationResponse(
                success=True,
                message=f"Registered {len(images)} face images for {username}"
            )
        else:
            raise HTTPException(status_code=400, detail="Face registration failed")
            
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/face/verify", response_model=AuthenticationResponse)
async def verify_face(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    """Verify face against registered user"""
    try:
        img = await read_image_file(file)
        
        # Perform recognition
        recognized_name, confidence, distance = face_recognizer.recognize(img)
        
        # Check if matches claimed username
        success = (
            recognized_name and
            recognized_name.lower() == username.lower() and
            distance <= 0.6  # InsightFace threshold
        )
        
        return AuthenticationResponse(
            success=success,
            username=recognized_name,
            confidence=confidence,
            distance=distance,
            message="Face verified" if success else "Face verification failed"
        )
        
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Iris Recognition ==========
@app.post("/api/iris/register", response_model=RegistrationResponse)
async def register_iris(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Register iris images for a user"""
    try:
        if len(files) < 3:
            raise HTTPException(
                status_code=400,
                detail="Minimum 3 iris images required"
            )
        
        images = []
        for file in files:
            img = await read_image_file(file)
            images.append(img)
        
        success = iris_recognizer.register_user(username, images)
        
        if success:
            await db.update_user_biometric(username, "iris", True)
            return RegistrationResponse(
                success=True,
                message=f"Registered {len(images)} iris images for {username}"
            )
        else:
            raise HTTPException(status_code=400, detail="Iris registration failed")
            
    except Exception as e:
        logger.error(f"Iris registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/iris/verify", response_model=AuthenticationResponse)
async def verify_iris(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    """Verify iris against registered user"""
    try:
        img = await read_image_file(file)
        
        recognized_name, hamming_dist = iris_recognizer.recognize(img)
        
        success = (
            recognized_name and
            recognized_name.lower() == username.lower() and
            hamming_dist <= 0.35
        )
        
        return AuthenticationResponse(
            success=success,
            username=recognized_name,
            distance=hamming_dist,
            confidence=1.0 - hamming_dist if hamming_dist else 0.0,
            message="Iris verified" if success else "Iris verification failed"
        )
        
    except Exception as e:
        logger.error(f"Iris verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Fingerprint Recognition ==========
@app.post("/api/fingerprint/register", response_model=RegistrationResponse)
async def register_fingerprint(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Register fingerprint images for a user"""
    try:
        if len(files) < 3:
            raise HTTPException(
                status_code=400,
                detail="Minimum 3 fingerprint images required"
            )
        
        images = []
        for file in files:
            img = await read_image_file(file)
            images.append(img)
        
        success = fingerprint_recognizer.register_user(username, images)
        
        if success:
            await db.update_user_biometric(username, "fingerprint", True)
            return RegistrationResponse(
                success=True,
                message=f"Registered {len(images)} fingerprints for {username}"
            )
        else:
            raise HTTPException(status_code=400, detail="Fingerprint registration failed")
            
    except Exception as e:
        logger.error(f"Fingerprint registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fingerprint/verify", response_model=AuthenticationResponse)
async def verify_fingerprint(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    """Verify fingerprint against registered user"""
    try:
        img = await read_image_file(file)
        
        recognized_name, match_score = fingerprint_recognizer.recognize(img)
        
        success = (
            recognized_name and
            recognized_name.lower() == username.lower() and
            match_score >= 0.85
        )
        
        return AuthenticationResponse(
            success=success,
            username=recognized_name,
            confidence=match_score,
            message="Fingerprint verified" if success else "Fingerprint verification failed"
        )
        
    except Exception as e:
        logger.error(f"Fingerprint verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Complete Authentication ==========
@app.post("/api/authenticate/complete")
async def complete_authentication(username: str = Form(...)):
    """Check if all three biometric factors are verified"""
    try:
        user = await db.get_user(username)
        if not user:
            return {"authenticated": False, "message": "User not found"}
        
        all_verified = (
            user.get("face_registered") and
            user.get("iris_registered") and
            user.get("fingerprint_registered")
        )
        
        return {
            "authenticated": all_verified,
            "username": username,
            "factors": {
                "face": user.get("face_registered", False),
                "iris": user.get("iris_registered", False),
                "fingerprint": user.get("fingerprint_registered", False)
            },
            "message": "All factors verified" if all_verified else "Incomplete verification"
        }
        
    except Exception as e:
        logger.error(f"Complete authentication error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

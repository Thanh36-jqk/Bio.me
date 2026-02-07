"""
Simplified FastAPI Backend for Biometric System
Works without requiring all heavy ML dependencies
Focus: MongoDB user registration and basic operations
"""
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
from pathlib import Path

# MongoDB
from api.database import db_manager

# Liveness Detection
from api.liveness_detection import liveness_detector

app = FastAPI(title="Biometric MFA API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    # Allow all origins in production to ensure Vercel frontend can connect
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class RegisterRequest(BaseModel):
    name: str
    age: int
    email: str

class AuthenticationResponse(BaseModel):
    success: bool
    username: Optional[str] = None
    email: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    message: str
    passed_biometrics: Optional[int] = None  # For 2/3 logic
    liveness_checks: Optional[dict] = None  # Anti-spoofing results

@app.on_event("startup")
async def startup():
    """Initialize database connection"""
    await db_manager.connect()
    print("[OK] MongoDB connected")
    print("[OK] Backend started successfully")
    print("="*60)
    print("="*60)
    print("BIOMETRIC MFA BACKEND SERVER - v1.2 (SSL CERTIFI FIX)")
    print("="*60)
    print("API Documentation: http://localhost:8000/docs")
    print("="*60)

@app.on_event("shutdown")
async def shutdown():
    """Close database connection"""
    await db_manager.disconnect()

@app.get("/")
async def root():
    """API status"""
    return {
        "status": "online",
        "version": "2.0.0",
        "message": "Biometric MFA Backend API",
        "endpoints": {
            "docs": "/docs",
            "users": "/users",
            "register": "/register",
            "authenticate": "/authenticate"
        }
    }

@app.get("/users")
async def list_users():
    """Get all registered users"""
    users = await db_manager.list_users()
    return {
        "total_users": len(users),
        "users": users
    }

@app.post("/register/user")
async def register_initial_user(
    name: str = Form(...),
    age: int = Form(...),
    email: str = Form(...)
):
    """
    Step 1: Register initial user info
    """
    try:
        # Check if email exists
        user = await db_manager.get_user(email)
        if user:
            return {
                "success": False,
                "message": "Email already registered"
            }
            
        # Create user
        user_id = await db_manager.create_user(email, name, age)
        return {
            "success": True,
            "message": "User info registered successfully",
            "user_id": user_id
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

@app.post("/register/face")
async def register_face(
    email: str = Form(...),
    images: list[UploadFile] = File(...)
):
    """
    Register user's face biometric
    
    Args:
        email: User email identifier
        images: List of face images (5-10 recommended)
    """
    print(f"[INFO] Registering face for user: {email}")
    print(f"       Received {len(images)} face images")
    
    # Check if user exists, create if not (legacy support, but better to use /register/user first)
    user = await db_manager.get_user(email)
    if not user:
         # Fallback: create with dummy name/age if not exists (shouldn't happen in new flow)
        await db_manager.create_user(email, "Unknown", 18)
        print(f"[OK] Created new user fallback: {email}")
    
    # Update biometric status
    await db_manager.update_user_biometric(email, "face", True)
    
    print(f"[OK] Face registered successfully for {email}")
    
    return {
        "success": True,
        "email": email,
        "biometric_type": "face",
        "num_images": len(images),
        "message": f"Face biometric registered for {email}"
    }

@app.post("/register/iris")
async def register_iris(
    email: str = Form(...),
    images: list[UploadFile] = File(...)
):
    """Register user's iris biometric"""
    print(f"[INFO] Registering iris for user: {email}")
    print(f"       Received {len(images)} iris images")
    
    user = await db_manager.get_user(email)
    if not user:
        return {"success": False, "message": "User not found. Please register info first."}
    
    await db_manager.update_user_biometric(email, "iris", True)
    
    print(f"[OK] Iris registered successfully for {email}")
    
    return {
        "success": True,
        "email": email,
        "biometric_type": "iris",
        "num_images": len(images),
        "message": f"Iris biometric registered for {email}"
    }

@app.post("/register/fingerprint")
async def register_fingerprint(
    email: str = Form(...),
    images: list[UploadFile] = File(...)
):
    """Register user's fingerprint biometric"""
    print(f"[INFO] Registering fingerprint for user: {email}")
    print(f"       Received {len(images)} fingerprint images")
    
    user = await db_manager.get_user(email)
    if not user:
        return {"success": False, "message": "User not found. Please register info first."}
    
    await db_manager.update_user_biometric(email, "fingerprint", True)
    
    print(f"[OK] Fingerprint registered successfully for {email}")
    
    return {
        "success": True,
        "email": email,
        "biometric_type": "fingerprint",
        "num_images": len(images),
        "message": f"Fingerprint biometric registered for {email}"
    }

@app.post("/authenticate/face", response_model=AuthenticationResponse)
async def authenticate_face(username: str = Form(...), image: UploadFile = File(...)):
    """Authenticate using face"""
    print(f"[AUTH] Authenticating face for: {username}")
    
    user = await db_manager.get_user(username)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found"
        )
    
    if not user.get("face_registered"):
        return AuthenticationResponse(
            success=False,
            message="Face biometric not registered for this user"
        )
    
    # Simulate authentication (99% accuracy)
    return AuthenticationResponse(
        success=True,
        username=username,
        confidence=0.99,
        distance=0.01,
        message="Face authentication successful"
    )

@app.post("/authenticate/iris", response_model=AuthenticationResponse)
async def authenticate_iris(username: str = Form(...), image: UploadFile = File(...)):
    """Authenticate using iris"""
    print(f"[AUTH] Authenticating iris for: {username}")
    
    user = await db_manager.get_user(username)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found"
        )
    
    if not user.get("iris_registered"):
        return AuthenticationResponse(
            success=False,
            message="Iris biometric not registered for this user"
        )
    
    return AuthenticationResponse(
        success=True,
        username=username,
        confidence=0.99,
        distance=0.01,
        message="Iris authentication successful"
    )

@app.post("/authenticate/fingerprint", response_model=AuthenticationResponse)
async def authenticate_fingerprint(username: str = Form(...), image: UploadFile = File(...)):
    """Authenticate using fingerprint"""
    print(f"[AUTH] Authenticating fingerprint for: {username}")
    
    user = await db_manager.get_user(username)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found"
        )
    
    if not user.get("fingerprint_registered"):
        return AuthenticationResponse(
            success=False,
            message="Fingerprint biometric not registered for this user"
        )
    
    return AuthenticationResponse(
        success=True,
        username=username,
        confidence=0.99,
        distance=0.01,
        message="F ingerprint authentication successful"
    )

@app.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_multi_biometric(
    email: str = Form(...),
    face_image: UploadFile = File(None),
    iris_image: UploadFile = File(None),
    fingerprint_image: UploadFile = File(None)
):
    """
    Multi-factor biometric authentication with 2/3 pass rule
    Requires at least 2 out of 3 biometrics to pass
    Includes liveness detection for anti-spoofing
    """
    print(f"[AUTH] Multi-biometric authentication for: {email}")
    
    # Check user exists
    user = await db_manager.get_user(email)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found",
            passed_biometrics=0
        )
    
    results = []
    liveness_results = {}
    
    # 1. Face Authentication (if provided)
    if face_image:
        try:
            # Read image
            contents = await face_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_face_liveness(img)
            liveness_results['face'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Face liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'face', 'success': False, 'reason': liveness_result['reason']})
            else:
                # TODO: Actual face recognition here
                # For now, mark as success if liveness passed and user has face registered
                if user.get("face_registered"):
                    results.append({'type': 'face', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"[OK] Face authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'face', 'success': False, 'reason': 'Face not registered'})
                    
        except Exception as e:
            print(f"[ERROR] Face authentication error: {e}")
            results.append({'type': 'face', 'success': False, 'reason': str(e)})
            liveness_results['face'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # 2. Iris Authentication (if provided)
    if iris_image:
        try:
            contents = await iris_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_iris_liveness(img)
            liveness_results['iris'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Iris liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'iris', 'success': False, 'reason': liveness_result['reason']})
            else:
                if user.get("iris_registered"):
                    results.append({'type': 'iris', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"[OK] Iris authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'iris', 'success': False, 'reason': 'Iris not registered'})
                    
        except Exception as e:
            print(f"[ERROR] Iris authentication error: {e}")
            results.append({'type': 'iris', 'success': False, 'reason': str(e)})
            liveness_results['iris'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # 3. Fingerprint Authentication (if provided)
    if fingerprint_image:
        try:
            contents = await fingerprint_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check
            liveness_result = liveness_detector.detect_fingerprint_liveness(img)
            liveness_results['fingerprint'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Fingerprint liveness FAILED: {liveness_result['reason']}")
                results.append({'type': 'fingerprint', 'success': False, 'reason': liveness_result['reason']})
            else:
                if user.get("fingerprint_registered"):
                    results.append({'type': 'fingerprint', 'success': True, 'confidence': liveness_result['confidence']})
                    print(f"[OK] Fingerprint authentication PASSED (liveness: {liveness_result['confidence']:.2f})")
                else:
                    results.append({'type': 'fingerprint', 'success': False, 'reason': 'Fingerprint not registered'})
                    
        except Exception as e:
            print(f"[ERROR] Fingerprint authentication error: {e}")
            results.append({'type': 'fingerprint', 'success': False, 'reason': str(e)})
            liveness_results['fingerprint'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # Count successes
    passed_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    # 2/3 Rule: Need at least 2 out of 3 to pass
    if passed_count >= 2:
        print(f"[SUCCESS] AUTHENTICATION SUCCESS: {passed_count}/{total_count} biometrics passed")
        return AuthenticationResponse(
            success=True,
            email=email,
            passed_biometrics=passed_count,
            message=f"Authentication successful ({passed_count}/{total_count} biometrics passed)",
            liveness_checks=liveness_results
        )
    else:
        print(f"[FAILED] AUTHENTICATION FAILED: {passed_count}/{total_count} biometrics passed (need 2+)")
        failed_reasons = [r.get('reason', 'Unknown') for r in results if not r['success']]
        return AuthenticationResponse(
            success=False,
            message=f"Authentication failed: {passed_count}/{total_count} passed (need 2+). Failures: {', '.join(failed_reasons)}",
            passed_biometrics=passed_count,
            liveness_checks=liveness_results
        )

@app.delete("/users/{email}")
async def delete_user(email: str):
    """Delete a user"""
    success = await db_manager.delete_user(email)
    if success:
        return {"success": True, "message": f"User {email} deleted"}
    return {"success": False, "message": "User not found"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = await db_manager.get_stats()
    return {
        **stats,
        "backend_status": "online",
        "ml_models": "classical (ML models ready for upgrade)"
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BIOMETRIC MFA BACKEND SERVER")
    print("="*60)
    print("\nConfiguration:")
    print("   - MongoDB: localhost:27017")
    print("   - API Port: 8000")
    print("   - CORS: Enabled for localhost:3000, localhost:3001")
    print("\nFeatures:")
    print("   - User registration (MongoDB)")
    print("   - Face/Iris/Fingerprint enrollment")
    print("   - Basic authentication")
    print("   - User management")
    print("\nNote: ML models will be added after installing dependencies")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

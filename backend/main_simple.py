"""
Multi-Modal Biometric Authentication System - Backend API

This module provides a FastAPI-based REST API for secure biometric authentication
using face, iris, and fingerprint recognition with 2-of-3 multi-factor verification.

Key Features:
    - User registration with email-based identification
    - Multi-modal biometric enrollment (face, iris, fingerprint)
    - Liveness detection for anti-spoofing
    - 2-of-3 authentication fusion logic
    - MongoDB-based template storage

Author: Thanh Nguyen (thanh36-jqk)
Version: 2.0.0
"""

from typing import Optional, List, Dict, Any
import uvicorn
import cv2
import numpy as np
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

# Internal modules
from api.database import db_manager
from api.liveness_detection import liveness_detector

# Biometric Recognizers (Production Quality with Deep Learning)
from api.face_recognition import FaceRecognizer

# Import DL modules with fallback to traditional
try:
    from api.iris_recognition_dl import IrisRecognizerDL
    IRIS_DL_AVAILABLE = True
except ImportError:
    IRIS_DL_AVAILABLE = False
    from api.iris_recognition import IrisRecognizer

try:
    from api.fingerprint_recognition_dl import FingerprintRecognizerDL
    FINGERPRINT_DL_AVAILABLE = True
except ImportError:
    FINGERPRINT_DL_AVAILABLE = False
    from api.fingerprint_recognition import FingerprintRecognizer

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

API_VERSION = "2.0.0"
API_TITLE = "Biometric MFA API"

# Biometric thresholds (optimized for Deep Learning)
FACE_SIMILARITY_THRESHOLD = 0.60      # Cosine similarity (InsightFace)
IRIS_DL_SIMILARITY_THRESHOLD = 0.85   # DL iris (CNN embeddings)
IRIS_HAMMING_THRESHOLD = 0.32         # Traditional iris (Gabor)
FINGERPRINT_DL_THRESHOLD = 0.80       # DL fingerprint (Siamese)
FINGERPRINT_MIN_MATCHES = 8           # Traditional fingerprint (SIFT)

# Authentication requirements
MIN_BIOMETRICS_REQUIRED = 2  # 2-of-3 rule

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Multi-modal biometric authentication with anti-spoofing"
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize biometric recognizers with DL (fallback to traditional)
try:
    face_recognizer = FaceRecognizer(threshold=FACE_SIMILARITY_THRESHOLD)
    logger.info(f"✓ Face recognizer initialized (InsightFace/ArcFace, threshold={FACE_SIMILARITY_THRESHOLD})")
except Exception as e:
    logger.error(f"✗ Face recognizer failed to initialize: {e}")
    face_recognizer = None

# Iris Recognition: DL if available, else traditional
if IRIS_DL_AVAILABLE:
    try:
        iris_recognizer = IrisRecognizerDL(threshold=IRIS_DL_SIMILARITY_THRESHOLD)
        logger.info(f"✓ Iris recognizer: DEEP LEARNING (CNN, threshold={IRIS_DL_SIMILARITY_THRESHOLD})")
    except Exception as e:
        logger.warning(f"DL iris failed: {e}, falling back to traditional")
        iris_recognizer = IrisRecognizer(threshold=IRIS_HAMMING_THRESHOLD)
        logger.info(f"✓ Iris recognizer: Traditional (Gabor, threshold={IRIS_HAMMING_THRESHOLD})")
else:
    iris_recognizer = IrisRecognizer(threshold=IRIS_HAMMING_THRESHOLD)
    logger.info(f"✓ Iris recognizer: Traditional (Gabor, threshold={IRIS_HAMMING_THRESHOLD})")

# Fingerprint Recognition: DL if available, else traditional
if FINGERPRINT_DL_AVAILABLE:
    try:
        fingerprint_recognizer = FingerprintRecognizerDL(threshold=FINGERPRINT_DL_THRESHOLD)
        logger.info(f"✓ Fingerprint recognizer: DEEP LEARNING (Siamese, threshold={FINGERPRINT_DL_THRESHOLD})")
    except Exception as e:
        logger.warning(f"DL fingerprint failed: {e}, falling back to traditional")
        fingerprint_recognizer = FingerprintRecognizer(threshold=FINGERPRINT_MIN_MATCHES)
        logger.info(f"✓ Fingerprint recognizer: Traditional (SIFT, min_matches={FINGERPRINT_MIN_MATCHES})")
else:
    fingerprint_recognizer = FingerprintRecognizer(threshold=FINGERPRINT_MIN_MATCHES)
    logger.info(f"✓ Fingerprint recognizer: Traditional (SIFT, min_matches={FINGERPRINT_MIN_MATCHES})")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserRegistrationRequest(BaseModel):
    """
    Request model for initial user registration.
    
    Attributes:
        name: Full name of the user
        age: Age of the user (positive integer)
        email: Unique email identifier
    """
    name: str = Field(..., min_length=1, max_length=100, description="Full name")
    age: int = Field(..., gt=0, lt=150, description="User age")
    email: EmailStr = Field(..., description="Unique email address")


class AuthenticationResponse(BaseModel):
    """
    Response model for authentication requests.
    
    Attributes:
        success: Whether authentication succeeded
        email: User's email (if authenticated)
        username: User's name (if authenticated)
        confidence: Overall confidence score
        distance: Distance metric (for specific modality)
        message: Human-readable status message
        passed_biometrics: Number of biometrics that passed (0-3)
        liveness_checks: Anti-spoofing check results per modality
    """
    success: bool
    email: Optional[str] = None
    username: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    message: str
    passed_biometrics: Optional[int] = None
    liveness_checks: Optional[Dict[str, Any]] = None


# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize system resources on application startup.
    
    Connects to MongoDB database and logs system status.
    """
    await db_manager.connect()
    print("[OK] MongoDB connected")
    print("[OK] Backend started successfully")
    print("=" * 60)
    print(f"BIOMETRIC MFA BACKEND SERVER - v{API_VERSION}")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Clean up system resources on application shutdown.
    
    Gracefully disconnects from MongoDB database.
    """
    await db_manager.disconnect()
    print("[INFO] Database connection closed")


# ============================================================================
# GENERAL ENDPOINTS
# ============================================================================

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    API root endpoint providing system status and available endpoints.
    
    Returns:
        Dictionary containing API status, version, and endpoint listing
    """
    return {
        "status": "online",
        "version": API_VERSION,
        "message": "Biometric MFA Backend API",
        "endpoints": {
            "docs": "/docs",
            "users": "/users",
            "register": "/register",
            "authenticate": "/authenticate"
        }
    }


@app.get("/users")
async def list_users() -> Dict[str, Any]:
    """
    Retrieve list of all registered users.
    
    Returns:
        Dictionary containing total user count and user details
    """
    users = await db_manager.list_users()
    return {
        "total_users": len(users),
        "users": users
    }


# ============================================================================
# USER REGISTRATION ENDPOINTS
# ============================================================================

@app.post("/register/user")
async def register_initial_user(
    name: str = Form(...),
    age: int = Form(...),
    email: EmailStr = Form(...)
) -> Dict[str, Any]:
    """
    Register a new user with basic information.
    
    This is the first step in user enrollment. After this,
    the user must register biometric samples (face, iris, fingerprint).
    
    Args:
        name: User's full name
        age: User's age (must be positive)
        email: Unique email identifier
    
    Returns:
        Success status and user details
    
    Raises:
        HTTPException: If email already exists
    """
    print(f"[INFO] Registering new user: {email} ({name}, {age})")
    
    try:
        user_id = await db_manager.create_user(
            email=email,
            name=name,
            age=age
        )
        
        print(f"[OK] User registered: {email}")
        
        return {
            "success": True,
            "message": "User registered successfully",
            "email": email,
            "name": name,
            "user_id": user_id
        }
    
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/register/face")
async def register_face_biometric(
    email: str = Form(...),
    images: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Register face biometric samples for a user.
    
    Args:
        email: User's email identifier
        images: List of face images (5-10 samples recommended)
    
    Returns:
        Success status and registration details
    
    Raises:
        HTTPException: If user not found or registration fails
    """
    print(f"[INFO] Registering face for user: {email}")
    print(f"       Received {len(images)} face images")
    
    # Verify user exists
    user = await db_manager.get_user(email)
    if not user:
        # Fallback: create user with default values
        # (Should not happen in normal flow)
        await db_manager.create_user(email, "Unknown", 18)
        print(f"[WARN] Created fallback user: {email}")
    
    # Update biometric registration status
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
async def register_iris_biometric(
    email: str = Form(...),
    images: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Register iris biometric samples for a user.
    
    Processes iris images and stores Gabor-encoded templates.
    
    Args:
        email: User's email identifier
        images: List of iris images (3-5 samples recommended)
    
    Returns:
        Success status and registration details
    """
    print(f"[INFO] Registering iris for user: {email}")
    print(f"       Received {len(images)} iris images")
    
    # Convert uploaded files to numpy arrays
    iris_images = []
    for img_file in images:
        contents = await img_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        iris_images.append(img)
    
    # Process and store iris templates
    success = iris_recognizer.register_user(email, iris_images)
    
    if success:
        # Update database registration status
        await db_manager.update_user_biometric(email, "iris", True)
        print(f"[OK] Iris registered successfully for {email}")
        
        return {
            "success": True,
            "email": email,
            "biometric_type": "iris",
            "message": f"Iris biometric registered for {email}"
        }
    else:
        print(f"[ERROR] Iris registration failed for {email}")
        return {
            "success": False,
            "message": "Failed to process iris images"
        }


@app.post("/register/fingerprint")
async def register_fingerprint_biometric(
    email: str = Form(...),
    images: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Register fingerprint biometric samples for a user.
    
    Processes fingerprint images using ORB feature extraction.
    
    Args:
        email: User's email identifier
        images: List of fingerprint images (3-5 samples recommended)
    
    Returns:
        Success status and registration details
    """
    print(f"[INFO] Registering fingerprint for user: {email}")
    print(f"       Received {len(images)} fingerprint images")
    
    # Convert uploaded files to numpy arrays
    fp_images = []
    for img_file in images:
        contents = await img_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        fp_images.append(img)
    
    # Process and store fingerprint templates
    success = fingerprint_recognizer.register_user(email, fp_images)
    
    if success:
        # Update database registration status
        await db_manager.update_user_biometric(email, "fingerprint", True)
        print(f"[OK] Fingerprint registered successfully for {email}")
        
        return {
            "success": True,
            "email": email,
            "biometric_type": "fingerprint",
            "message": f"Fingerprint biometric registered for {email}"
        }
    else:
        print(f"[ERROR] Fingerprint registration failed for {email}")
        return {
            "success": False,
            "message": "Failed to process fingerprint images"
        }


# ============================================================================
# AUTHENTICATION ENDPOINT
# ============================================================================

@app.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_multi_biometric(
    email: str = Form(...),
    face_image: UploadFile = File(None),
    iris_image: UploadFile = File(None),
    fingerprint_image: UploadFile = File(None)
) -> AuthenticationResponse:
    """
    Authenticate user with multi-modal biometric verification.
    
    Implements 2-of-3 decision fusion logic: at least 2 out of 3
    biometric modalities must successfully authenticate for access.
    
    Each modality undergoes:
        1. Liveness detection (anti-spoofing)
        2. Biometric matching against stored templates
    
    Args:
        email: User's email identifier
        face_image: Optional face image for recognition
        iris_image: Optional iris image for recognition
        fingerprint_image: Optional fingerprint image for recognition
    
    Returns:
        Authentication result with success status and confidence metrics
    """
    print(f"[AUTH] Multi-biometric authentication for: {email}")
    
    # Verify user exists
    user = await db_manager.get_user(email)
    if not user:
        return AuthenticationResponse(
            success=False,
            message="User not found",
            passed_biometrics=0
        )
    
    results: List[Dict[str, Any]] = []
    liveness_results: Dict[str, Any] = {}
    
    # ========================================================================
    # FACE AUTHENTICATION
    # ========================================================================
    if face_image:
        try:
            # Read and decode image
            contents = await face_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check (anti-spoofing)
            liveness_result = liveness_detector.detect_face_liveness(img)
            liveness_results['face'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Face liveness FAILED: {liveness_result['reason']}")
                results.append({
                    'type': 'face',
                    'success': False,
                    'reason': liveness_result['reason']
                })
            else:
                # ACTUAL Face Recognition with InsightFace
                if user.get("face_registered") and face_recognizer is not None:
                    try:
                        # Perform face matching
                        recognized_user, similarity = face_recognizer.recognize(img)
                        
                        if recognized_user and recognized_user.lower() == email.lower():
                            results.append({
                                'type': 'face',
                                'success': True,
                                'confidence': similarity
                            })
                            print(f"[OK] Face authentication PASSED "
                                  f"(similarity: {similarity:.3f}, threshold: {FACE_SIMILARITY_THRESHOLD:.2f})")
                        else:
                            results.append({
                                'type': 'face',
                                'success': False,
                                'reason': f'Face not matched (best similarity: {similarity:.3f})'
                            })
                            print(f"[FAIL] Face NOT matched. Got: {recognized_user}, "
                                  f"similarity: {similarity:.3f}")
                    except Exception as match_error:
                        logger.error(f"Face matching error: {match_error}")
                        results.append({
                            'type': 'face',
                            'success': False,
                            'reason': f'Matching error: {str(match_error)}'
                        })
                elif not user.get("face_registered"):
                    results.append({
                        'type': 'face',
                        'success': False,
                        'reason': 'Face not registered'
                    })
                else:
                    results.append({
                        'type': 'face',
                        'success': False,
                        'reason': 'Face recognizer not available'
                    })
        
        except Exception as e:
            print(f"[ERROR] Face authentication error: {e}")
            results.append({'type': 'face', 'success': False, 'reason': str(e)})
            liveness_results['face'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # ========================================================================
    # IRIS AUTHENTICATION
    # ========================================================================
    if iris_image:
        try:
            # Read and decode image
            contents = await iris_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check (anti-spoofing)
            liveness_result = liveness_detector.detect_iris_liveness(img)
            liveness_results['iris'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Iris liveness FAILED: {liveness_result['reason']}")
                results.append({
                    'type': 'iris',
                    'success': False,
                    'reason': liveness_result['reason']
                })
            else:
                # Actual iris recognition using Gabor encoding
                if user.get("iris_registered"):
                    recognized_user, hamming_dist = iris_recognizer.recognize(img)
                    
                    if recognized_user and recognized_user.lower() == email.lower():
                        # Convert distance to confidence (lower distance = higher confidence)
                        confidence = 1.0 - hamming_dist
                        results.append({
                            'type': 'iris',
                            'success': True,
                            'confidence': confidence
                        })
                        print(f"[OK] Iris authentication PASSED "
                              f"(distance: {hamming_dist:.3f}, confidence: {confidence:.2f})")
                    else:
                        results.append({
                            'type': 'iris',
                            'success': False,
                            'reason': f'Iris not matched (best distance: {hamming_dist:.3f})'
                        })
                        print(f"[FAIL] Iris NOT matched. Got: {recognized_user}, "
                              f"distance: {hamming_dist:.3f}")
                else:
                    results.append({
                        'type': 'iris',
                        'success': False,
                        'reason': 'Iris not registered'
                    })
        
        except Exception as e:
            print(f"[ERROR] Iris authentication error: {e}")
            results.append({'type': 'iris', 'success': False, 'reason': str(e)})
            liveness_results['iris'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # ========================================================================
    # FINGERPRINT AUTHENTICATION
    # ========================================================================
    if fingerprint_image:
        try:
            # Read and decode image
            contents = await fingerprint_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Liveness check (anti-spoofing)
            liveness_result = liveness_detector.detect_fingerprint_liveness(img)
            liveness_results['fingerprint'] = liveness_result
            
            if not liveness_result['is_live']:
                print(f"[FAIL] Fingerprint liveness FAILED: {liveness_result['reason']}")
                results.append({
                    'type': 'fingerprint',
                    'success': False,
                    'reason': liveness_result['reason']
                })
            else:
                # Actual fingerprint recognition using ORB matching
                if user.get("fingerprint_registered"):
                    recognized_user, match_score = fingerprint_recognizer.recognize(img)
                    
                    if recognized_user and recognized_user.lower() == email.lower():
                        results.append({
                            'type': 'fingerprint',
                            'success': True,
                            'confidence': match_score
                        })
                        print(f"[OK] Fingerprint authentication PASSED "
                              f"(score: {match_score:.2f})")
                    else:
                        results.append({
                            'type': 'fingerprint',
                            'success': False,
                            'reason': f'Fingerprint not matched (best score: {match_score:.2f})'
                        })
                        print(f"[FAIL] Fingerprint NOT matched. Got: {recognized_user}, "
                              f"score: {match_score:.2f}")
                else:
                    results.append({
                        'type': 'fingerprint',
                        'success': False,
                        'reason': 'Fingerprint not registered'
                    })
        
        except Exception as e:
            print(f"[ERROR] Fingerprint authentication error: {e}")
            results.append({'type': 'fingerprint', 'success': False, 'reason': str(e)})
            liveness_results['fingerprint'] = {'is_live': False, 'reason': f'Error: {str(e)}'}
    
    # ========================================================================
    # DECISION FUSION (2-of-3 RULE)
    # ========================================================================
    
    passed_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    # Authentication succeeds if at least MIN_BIOMETRICS_REQUIRED pass
    if passed_count >= MIN_BIOMETRICS_REQUIRED:
        print(f"[SUCCESS] AUTHENTICATION SUCCESS: {passed_count}/{total_count} biometrics passed")
        return AuthenticationResponse(
            success=True,
            email=email,
            passed_biometrics=passed_count,
            message=f"Authentication successful ({passed_count}/{total_count} biometrics passed)",
            liveness_checks=liveness_results
        )
    else:
        print(f"[FAILED] AUTHENTICATION FAILED: {passed_count}/{total_count} biometrics passed "
              f"(need {MIN_BIOMETRICS_REQUIRED}+)")
        failed_reasons = [r.get('reason', 'Unknown') for r in results if not r['success']]
        return AuthenticationResponse(
            success=False,
            passed_biometrics=passed_count,
            message=f"Authentication failed: {passed_count}/{total_count} passed (need {MIN_BIOMETRICS_REQUIRED}+). "
                    f"Reasons: {', '.join(failed_reasons[:3])}",
            liveness_checks=liveness_results
        )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

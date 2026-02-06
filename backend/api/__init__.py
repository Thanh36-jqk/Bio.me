# API Module
# Commented out to avoid InsightFace dependency for now
# from .face_recognition import FaceRecognizer
# from .iris_recognition import IrisRecognizer
# from .fingerprint_recognition import FingerprintRecognizer
from .database import db_manager

__all__ = ['db_manager']

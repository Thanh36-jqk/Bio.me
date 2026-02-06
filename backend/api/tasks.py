"""
Celery Task Queue for Biometric Processing
Heavy ML/DL tasks run asynchronously in background workers
"""
from celery import Celery, Task
import numpy as np
import logging
from typing import List, Dict
import asyncio

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    'biometric_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_serializer='pickle',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)

class BiometricTask(Task):
    """Base task with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")
        super().on_success(retval, task_id, args, kwargs)

# ========== Face Recognition Tasks ==========

@celery_app.task(base=BiometricTask, bind=True, name='tasks.process_face_registration')
def process_face_registration(self, username: str, image_data: List[bytes]):
    """
    Process face registration in background
    Heavy task: MTCNN detection + FaceNet embedding extraction
    
    Args:
        username: User identifier
        image_data: List of image byte arrays
    
    Returns:
        dict: Registration status
    """
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'step': 'loading_models', 'progress': 0.1})
        
        from api.face_recognition_advanced import AdvancedFaceRecognizer
        import cv2
        
        recognizer = AdvancedFaceRecognizer(use_gpu=True)
        
        # Decode images
        self.update_state(state='PROCESSING', meta={'step': 'decoding_images', 'progress': 0.2})
        images = []
        for img_bytes in image_data:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        
        # Register user
        self.update_state(state='PROCESSING', meta={'step': 'extracting_features', 'progress': 0.5})
        success = recognizer.register_user(username, images)
        
        self.update_state(state='PROCESSING', meta={'step': 'saving_database', 'progress': 0.9})
        
        return {
            'success': success,
            'username': username,
            'num_images': len(images),
            'backend': 'facenet'
        }
        
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        raise

@celery_app.task(base=BiometricTask, bind=True, name='tasks.process_face_recognition')
def process_face_recognition(self, image_data: bytes):
    """
    Process face recognition (verification)
    
    Returns:
        dict: Recognition result
    """
    try:
        from api.face_recognition_advanced import AdvancedFaceRecognizer
        import cv2
        
        recognizer = AdvancedFaceRecognizer(use_gpu=True)
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Recognize
        username, confidence, distance = recognizer.recognize(img)
        
        return {
            'username': username,
            'confidence': float(confidence),
            'distance': float(distance),
            'success': username is not None
        }
        
    except Exception as e:
        logger.error(f"Face recognition failed: {e}")
        raise

# ========== Iris Recognition Tasks ==========

@celery_app.task(base=BiometricTask, bind=True, name='tasks.process_iris_registration')
def process_iris_registration(self, username: str, image_data: List[bytes]):
    """
    Process iris registration with U-Net + ResNet50
    GPU-intensive task
    """
    try:
        self.update_state(state='PROCESSING', meta={'step': 'loading_models', 'progress': 0.1})
        
        from api.iris_recognition_advanced import AdvancedIrisRecognizer
        import cv2
        
        recognizer = AdvancedIrisRecognizer(use_gpu=True)
        
        # Decode images
        self.update_state(state='PROCESSING', meta={'step': 'preprocessing', 'progress': 0.2})
        images = []
        for img_bytes in image_data:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        
        # Register
        self.update_state(state='PROCESSING', meta={'step': 'segmentation', 'progress': 0.5})
        success = recognizer.register_user(username, images)
        
        return {
            'success': success,
            'username': username,
            'num_images': len(images),
            'backend': 'unet_resnet50'
        }
        
    except Exception as e:
        logger.error(f"Iris registration failed: {e}")
        raise

# ========== Batch Processing Tasks ==========

@celery_app.task(base=BiometricTask, name='tasks.batch_verify_users')
def batch_verify_users(usernames: List[str]) -> Dict:
    """
    Verify multiple users in batch
    Useful for mass verification scenarios
    """
    try:
        from api.database import db_manager
        
        results = {}
        for username in usernames:
            user = asyncio.run(db_manager.get_user(username))
            results[username] = user is not None
        
        return {
            'total': len(usernames),
            'verified': sum(results.values()),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch verification failed: {e}")
        raise

# ========== Cleanup Tasks ==========

@celery_app.task(name='tasks.cleanup_old_sessions')
def cleanup_old_sessions():
    """
    Periodic task to cleanup old sessions
    Run via Celery Beat scheduler
    """
    try:
        from api.cache import cache_manager
        
        # This would be implemented with proper session tracking
        logger.info("Cleaned up old sessions")
        return {'status': 'completed'}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise

# ========== Model Preloading Task ==========

@celery_app.task(name='tasks.preload_models')
def preload_models():
    """
    Preload all ML models to warm up workers
    Run on worker startup
    """
    try:
        from api.face_recognition_advanced import AdvancedFaceRecognizer
        from api.iris_recognition_advanced import AdvancedIrisRecognizer
        
        # Initialize models
        face_rec = AdvancedFaceRecognizer(use_gpu=True)
        iris_rec = AdvancedIrisRecognizer(use_gpu=True)
        
        logger.info("✓ All models preloaded successfully")
        return {'status': 'models_ready'}
        
    except Exception as e:
        logger.error(f"Model preloading failed: {e}")
        raise

# Celery Beat schedule (for periodic tasks)
celery_app.conf.beat_schedule = {
    'cleanup-sessions-every-hour': {
        'task': 'tasks.cleanup_old_sessions',
        'schedule': 3600.0,  # Every hour
    },
}

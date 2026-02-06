"""
Face Recognition Module using InsightFace (ArcFace)
State-of-the-art accuracy: 99%+
"""
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, model_name='buffalo_l', threshold=0.6):
        """
        Initialize InsightFace recognizer
        
        Args:
            model_name: InsightFace model ('buffalo_l' recommended)
            threshold: Distance threshold for recognition (default: 0.6)
        """
        self.threshold = threshold
        self.database_path = Path("models/face/face_database.pkl")
        self.database = {}
        
        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode
            logger.info(f"InsightFace model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise
        
        # Load existing database
        self._load_database()
    
    def _load_database(self):
        """Load face embeddings database from disk"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'rb') as f:
                    self.database = pickle.load(f)
                logger.info(f"Loaded {len(self.database)} users from database")
            except Exception as e:
                logger.warning(f"Could not load database: {e}")
                self.database = {}
    
    def _save_database(self):
        """Save face embeddings database to disk"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.database, f)
            logger.info("Database saved successfully")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def register_user(self, username: str, images: list) -> bool:
        """
        Register a user with multiple face images
        
        Args:
            username: User identifier
            images: List of face images (numpy arrays, BGR format)
        
        Returns:
            bool: Success status
        """
        embeddings = []
        
        for idx, img in enumerate(images):
            try:
                # Detect faces
                faces = self.app.get(img)
                
                if len(faces) == 0:
                    logger.warning(f"No face detected in image {idx} for {username}")
                    continue
                
                # Use largest face if multiple detected
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                
                # Extract embedding (512-dim vector)
                embedding = faces[0].embedding
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                continue
        
        if len(embeddings) == 0:
            logger.error(f"No valid embeddings extracted for {username}")
            return False
        
        # Store average embedding for robustness
        avg_embedding = np.mean(embeddings, axis=0)
        self.database[username.lower()] = {
            'embedding': avg_embedding,
            'num_samples': len(embeddings)
        }
        
        self._save_database()
        logger.info(f"Registered {username} with {len(embeddings)} face samples")
        return True
    
    def recognize(self, img: np.ndarray, return_all=False):
        """
        Recognize face in image
        
        Args:
            img: Input image (numpy array, BGR format)
            return_all: If True, return all matches above threshold
        
        Returns:
            tuple: (username, confidence, distance) or (None, 0, inf) if no match
        """
        try:
            # Detect faces
            faces = self.app.get(img)
            
            if len(faces) == 0:
                logger.warning("No face detected in query image")
                return None, 0.0, float('inf')
            
            # Use largest face
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            query_embedding = faces[0].embedding
            
            # Find best match
            best_username = None
            best_distance = float('inf')
            
            for username, data in self.database.items():
                db_embedding = data['embedding']
                
                # L2 distance
                distance = np.linalg.norm(query_embedding - db_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_username = username
            
            # Check threshold
            if best_distance <= self.threshold:
                confidence = 1.0 - (best_distance / 2.0)  # Normalize to 0-1
                confidence = max(0.0, min(1.0, confidence))
                return best_username, confidence, float(best_distance)
            else:
                return None, 0.0, float(best_distance)
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None, 0.0, float('inf')
    
    def delete_user(self, username: str) -> bool:
        """Delete user from database"""
        username_lower = username.lower()
        if username_lower in self.database:
            del self.database[username_lower]
            self._save_database()
            logger.info(f"Deleted user: {username}")
            return True
        return False
    
    def list_users(self) -> list:
        """Get list of registered users"""
        return list(self.database.keys())


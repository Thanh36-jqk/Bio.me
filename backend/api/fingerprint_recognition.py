"""
Fingerprint Recognition Module
Using CNN-based deep features (or fallback to ORB if model unavailable)
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Try to import deep learning fingerprint
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, using ORB fallback")

logger = logging.getLogger(__name__)

class FingerprintRecognizer:
    def __init__(self, threshold=0.85, use_deep_learning=False):
        """
        Initialize Fingerprint recognizer
        
        Args:
            threshold: Match threshold (0.85 for DL, 15 matches for ORB)
            use_deep_learning: Use CNN if available (experimental)
        """
        self.threshold = threshold
        self.use_dl = use_deep_learning and TF_AVAILABLE
        self.database_path = Path("models/fingerprint/fingerprint_database.pkl")
        self.database = {}
        
        if self.use_dl:
            self._init_deep_model()
        else:
            # Use ORB as proven fallback
            self.orb = cv2.ORB_create(nfeatures=1500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self._load_database()
    
    def _init_deep_model(self):
        """Initialize deep learning model (if available)"""
        model_path = Path("models/fingerprint/deepprint_model.h5")
        if model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(model_path))
                logger.info("Loaded deep fingerprint model")
            except Exception as e:
                logger.warning(f"Could not load DL model: {e}, using ORB")
                self.use_dl = False
                self.orb = cv2.ORB_create(nfeatures=1500)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            logger.warning("DL model not found, using ORB")
            self.use_dl = False
            self.orb = cv2.ORB_create(nfeatures=1500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _load_database(self):
        """Load fingerprint database"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'rb') as f:
                    self.database = pickle.load(f)
                logger.info(f"Loaded {len(self.database)} fingerprint users")
            except Exception as e:
                logger.warning(f"Could not load fingerprint database: {e}")
                self.database = {}
    
    def _save_database(self):
        """Save fingerprint database"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.database, f)
            logger.info("Fingerprint database saved")
        except Exception as e:
            logger.error(f"Failed to save fingerprint database: {e}")
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess fingerprint image
        
        Args:
            img: Input fingerprint image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Enhanced preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to enhance ridges
        enhanced = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        return enhanced
    
    def _extract_features_orb(self, img: np.ndarray):
        """Extract ORB features from fingerprint"""
        processed = self._preprocess(img)
        kp, des = self.orb.detectAndCompute(processed, None)
        return kp, des
    
    def _extract_features_dl(self, img: np.ndarray):
        """Extract deep learning features (if model available)"""
        if not self.use_dl:
            return None
        
        try:
            # Resize and normalize
            processed = cv2.resize(img, (224, 224))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=(0, -1))
            
            # Extract features
            features = self.model.predict(processed, verbose=0)[0]
            features = features / np.linalg.norm(features)  # L2 normalize
            
            return features
        except Exception as e:
            logger.error(f"DL feature extraction failed: {e}")
            return None
    
    def register_user(self, username: str, images: List[np.ndarray]) -> bool:
        """
        Register user with fingerprint images
        
        Args:
            username: User identifier
            images: List of fingerprint images
        
        Returns:
            bool: Success status
        """
        if self.use_dl:
            return self._register_user_dl(username, images)
        else:
            return self._register_user_orb(username, images)
    
    def _register_user_orb(self, username: str, images: List[np.ndarray]) -> bool:
        """Register using ORB features"""
        all_descriptors = []
        
        for idx, img in enumerate(images):
            kp, des = self._extract_features_orb(img)
            if des is not None and len(des) > 0:
                all_descriptors.append(des)
            else:
                logger.warning(f"No features in fingerprint {idx} for {username}")
        
        if len(all_descriptors) == 0:
            logger.error(f"No valid fingerprint features for {username}")
            return False
        
        # Concatenate all descriptors
        combined_des = np.vstack(all_descriptors)
        
        self.database[username.lower()] = {
            'type': 'orb',
            'descriptors': combined_des,
            'num_samples': len(images)
        }
        
        self._save_database()
        logger.info(f"Registered {username} with {len(all_descriptors)} fingerprint samples (ORB)")
        return True
    
    def _register_user_dl(self, username: str, images: List[np.ndarray]) -> bool:
        """Register using deep learning features"""
        features_list = []
        
        for idx, img in enumerate(images):
            feat = self._extract_features_dl(img)
            if feat is not None:
                features_list.append(feat)
            else:
                logger.warning(f"Could not extract DL features from {idx}")
        
        if len(features_list) == 0:
            logger.error(f"No valid DL features for {username}")
            return False
        
        avg_features = np.mean(features_list, axis=0)
        
        self.database[username.lower()] = {
            'type': 'deep',
            'features': avg_features,
            'num_samples': len(images)
        }
        
        self._save_database()
        logger.info(f"Registered {username} with {len(features_list)} fingerprints (DL)")
        return True
    
    def recognize(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize fingerprint in image
        
        Args:
            img: Input fingerprint image
        
        Returns:
            tuple: (username, confidence/match_score) or (None, 0)
        """
        if self.use_dl:
            return self._recognize_dl(img)
        else:
            return self._recognize_orb(img)
    
    def _recognize_orb(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize using ORB matching"""
        try:
            kp_query, des_query = self._extract_features_orb(img)
            
            if des_query is None or len(des_query) == 0:
                logger.warning("No features in query fingerprint")
                return None, 0.0
            
            best_username = None
            best_score = 0
            
            for username, data in self.database.items():
                if data.get('type') != 'orb':
                    continue
                
                des_db = data['descriptors']
                
                # Match using BFMatcher
                matches = self.matcher.knnMatch(des_query, des_db, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                num_matches = len(good_matches)
                
                if num_matches > best_score:
                    best_score = num_matches
                    best_username = username
            
            # Check threshold (minimum 15 matches for ORB)
            min_matches = 15
            if best_score >= min_matches:
                # Normalize score to 0-1 range
                confidence = min(1.0, best_score / 50.0)
                return best_username, confidence
            else:
                return None, 0.0
                
        except Exception as e:
            logger.error(f"ORB recognition error: {e}")
            return None, 0.0
    
    def _recognize_dl(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize using deep learning"""
        try:
            query_feat = self._extract_features_dl(img)
            
            if query_feat is None:
                return None, 0.0
            
            best_username = None
            best_score = -1.0
            
            for username, data in self.database.items():
                if data.get('type') != 'deep':
                    continue
                
                db_feat = data['features']
                
                # Cosine similarity
                score = np.dot(query_feat, db_feat)
                
                if score > best_score:
                    best_score = score
                    best_username = username
            
            if best_score >= self.threshold:
                return best_username, float(best_score)
            else:
                return None, float(best_score)
                
        except Exception as e:
            logger.error(f"DL recognition error: {e}")
            return None, 0.0
    
    def delete_user(self, username: str) -> bool:
        """Delete user from database"""
        username_lower = username.lower()
        if username_lower in self.database:
            del self.database[username_lower]
            self._save_database()
            return True
        return False
    
    def list_users(self) -> list:
        """Get list of registered users"""
        return list(self.database.keys())

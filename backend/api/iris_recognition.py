"""
Iris Recognition Module
Hybrid approach: CNN segmentation + Gabor encoding
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

# Import from existing iris module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "modules" / "iris"))
try:
    import iris as iris_classic
except:
    iris_classic = None

logger = logging.getLogger(__name__)

class IrisRecognizer:
    def __init__(self, threshold=0.35):
        """
        Initialize Iris recognizer
        
        Args:
            threshold: Hamming distance threshold (default: 0.35)
        """
        self.threshold = threshold
        self.database_path = Path("models/iris/iris_database.pkl")
        self.database = {}
        self._load_database()
    
    def _load_database(self):
        """Load iris codes database"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'rb') as f:
                    self.database = pickle.load(f)
                logger.info(f"Loaded {len(self.database)} iris users from database")
            except Exception as e:
                logger.warning(f"Could not load iris database: {e}")
                self.database = {}
    
    def _save_database(self):
        """Save iris codes database"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.database, f)
            logger.info("Iris database saved")
        except Exception as e:
            logger.error(f"Failed to save iris database: {e}")
    
    def _process_iris(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Process iris image and extract Gabor code
        
        Args:
            img: Input iris image (BGR or grayscale)
        
        Returns:
            Iris code (binary array) or None if failed
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Remove specular reflections
            if iris_classic:
                clean = iris_classic.remove_specular_reflections(gray)
            else:
                clean = gray
            
            # Detect iris boundaries
            if iris_classic:
                pupil, iris_boundary = iris_classic.detect_iris_boundaries(clean)
            else:
                # Fallback simple detection
                pupil, iris_boundary = self._simple_detect(clean)
            
            if pupil is None or iris_boundary is None:
                logger.warning("Failed to detect iris boundaries")
                return None
            
            # Normalize to polar coordinates
            if iris_classic:
                polar = iris_classic.normalize_iris(clean, pupil, iris_boundary, radials=64, angles=512)
            else:
                polar = self._simple_normalize(clean, pupil, iris_boundary)
            
            # Gabor encoding
            if iris_classic:
                code = iris_classic.gabor_encode(polar)
            else:
                code = self._simple_gabor(polar)
            
            return code
            
        except Exception as e:
            logger.error(f"Iris processing error: {e}")
            return None
    
    def _simple_detect(self, gray):
        """Simple fallback iris detection using Hough circles"""
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=80, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None and len(circles[0]) >= 2:
            circles = circles[0]
            # Assume smallest is pupil, largest is iris
            sorted_circles = sorted(circles, key=lambda c: c[2])
            pupil = tuple(sorted_circles[0].astype(int))
            iris = tuple(sorted_circles[-1].astype(int))
            return pupil, iris
        
        return None, None
    
    def _simple_normalize(self, img, pupil, iris, radials=64, angles=512):
        """Simple polar transformation"""
        x_p, y_p, r_p = pupil
        x_i, y_i, r_i = iris
        
        theta = np.linspace(0, 2*np.pi, angles, endpoint=False)
        r_norm = np.linspace(0, 1, radials)
        
        X = x_p + (r_p + r_norm[:, None] * (r_i - r_p)) * np.cos(theta)[None, :]
        Y = y_p + (r_p + r_norm[:, None] * (r_i - r_p)) * np.sin(theta)[None, :]
        
        X = np.clip(X, 0, img.shape[1]-1).astype(np.float32)
        Y = np.clip(Y, 0, img.shape[0]-1).astype(np.float32)
        
        polar = cv2.remap(img, X, Y, cv2.INTER_LINEAR)
        return polar
    
    def _simple_gabor(self, polar):
        """Simple Gabor encoding"""
        kernel = cv2.getGaborKernel((21, 21), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(polar.astype(np.float32), cv2.CV_32F, kernel)
        code = (filtered > np.mean(filtered)).astype(np.uint8)
        return code
    
    def _hamming_distance(self, code1: np.ndarray, code2: np.ndarray) -> float:
        """Calculate Hamming distance between two iris codes"""
        if code1.shape != code2.shape:
            return 1.0
        
        total_bits = code1.size
        diff = np.sum(code1 != code2)
        return diff / total_bits
    
    def register_user(self, username: str, images: list) -> bool:
        """
        Register user with iris images
        
        Args:
            username: User identifier
            images: List of iris images
        
        Returns:
            bool: Success status
        """
        codes = []
        
        for idx, img in enumerate(images):
            code = self._process_iris(img)
            if code is not None:
                codes.append(code)
            else:
                logger.warning(f"Could not process iris image {idx} for {username}")
        
        if len(codes) == 0:
            logger.error(f"No valid iris codes for {username}")
            return False
        
        # Store all codes (not averaged since they're binary)
        self.database[username.lower()] = {
            'codes': codes,
            'num_samples': len(codes)
        }
        
        self._save_database()
        logger.info(f"Registered {username} with {len(codes)} iris samples")
        return True
    
    def recognize(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize iris in image
        
        Args:
            img: Input iris image
        
        Returns:
            tuple: (username, hamming_distance) or (None, 1.0)
        """
        try:
            query_code = self._process_iris(img)
            
            if query_code is None:
                logger.warning("Failed to process query iris")
                return None, 1.0
            
            best_username = None
            best_distance = 1.0
            
            for username, data in self.database.items():
                codes = data['codes']
                
                # Compare with all stored codes, take minimum distance
                for code in codes:
                    distance = self._hamming_distance(query_code, code)
                    if distance < best_distance:
                        best_distance = distance
                        best_username = username
            
            if best_distance <= self.threshold:
                return best_username, best_distance
            else:
                return None, best_distance
                
        except Exception as e:
            logger.error(f"Iris recognition error: {e}")
            return None, 1.0
    
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

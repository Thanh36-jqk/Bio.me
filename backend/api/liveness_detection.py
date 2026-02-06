"""
Liveness Detection for Biometric Anti-Spoofing
Prevents attacks using photos, videos, or printed images
"""
import cv2
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LivenessDetector:
    """
    Multi-modal liveness detection
    - Face: Texture analysis + Blur detection
    - Iris: Quality checks + Resolution validation
    - Fingerprint: Edge sharpness analysis
    """
    
    # Thresholds (tuned for balance between security and usability)
    FACE_BLUR_THRESHOLD = 100.0  # Laplacian variance
    FACE_LBP_THRESHOLD = 50.0    # Local Binary Pattern variance
    IRIS_MIN_RESOLUTION = 200    # Minimum pixels
    IRIS_EDGE_THRESHOLD = 0.15   # Edge density
    FP_EDGE_THRESHOLD = 0.20     # Fingerprint edge quality
    
    def __init__(self):
        """Initialize liveness detector"""
        logger.info("Liveness Detector initialized")
    
    def detect_face_liveness(self, image: np.ndarray) -> Dict:
        """
        Detect if face image is from a real person or photo/screen
        
        Methods:
        1. Blur detection (photos are often blurrier)
        2. Texture analysis using LBP (Local Binary Patterns)
        3. Color distribution analysis
        
        Args:
            image: BGR image containing face
            
        Returns:
            {
                'is_live': bool,
                'confidence': float (0-1),
                'reason': str,
                'blur_score': float,
                'texture_score': float
            }
        """
        try:
            if image is None or image.size == 0:
                return {
                    'is_live': False,
                    'confidence': 0.0,
                    'reason': 'Invalid image',
                    'blur_score': 0.0,
                    'texture_score': 0.0
                }
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Blur Detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < self.FACE_BLUR_THRESHOLD:
                return {
                    'is_live': False,
                    'confidence': min(blur_score / self.FACE_BLUR_THRESHOLD, 1.0),
                    'reason': f'Image too blurry (score: {blur_score:.1f})',
                    'blur_score': float(blur_score),
                    'texture_score': 0.0
                }
            
            # 2. Texture Analysis using LBP
            lbp_variance = self._calculate_lbp_variance(gray)
            
            if lbp_variance < self.FACE_LBP_THRESHOLD:
                return {
                    'is_live': False,
                    'confidence': 0.4,
                    'reason': 'Detected photo/screen (low texture variance)',
                    'blur_score': float(blur_score),
                    'texture_score': float(lbp_variance)
                }
            
            # 3. Color distribution check (optional, for better accuracy)
            color_score = self._check_color_distribution(image)
            
            # Calculate overall confidence
            confidence = min((blur_score / self.FACE_BLUR_THRESHOLD) * 0.5 + 
                           (lbp_variance / self.FACE_LBP_THRESHOLD) * 0.3 +
                           color_score * 0.2, 1.0)
            
            logger.info(f"Face liveness: LIVE (blur={blur_score:.1f}, lbp={lbp_variance:.1f}, conf={confidence:.2f})")
            
            return {
                'is_live': True,
                'confidence': float(confidence),
                'reason': 'Live face detected',
                'blur_score': float(blur_score),
                'texture_score': float(lbp_variance)
            }
            
        except Exception as e:
            logger.error(f"Face liveness detection failed: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'reason': f'Detection error: {str(e)}',
                'blur_score': 0.0,
                'texture_score': 0.0
            }
    
    def detect_iris_liveness(self, image: np.ndarray) -> Dict:
        """
        Detect if iris image is authentic
        
        Methods:
        1. Resolution check
        2. Pupil detection
        3. Edge quality analysis
        
        Args:
            image: BGR/Grayscale image of eye/iris
            
        Returns:
            {
                'is_live': bool,
                'confidence': float,
                'reason': str
            }
        """
        try:
            if image is None or image.size == 0:
                return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid image'}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # 1. Resolution check
            if h < self.IRIS_MIN_RESOLUTION or w < self.IRIS_MIN_RESOLUTION:
                return {
                    'is_live': False,
                    'confidence': 0.2,
                    'reason': f'Resolution too low ({w}x{h}, need {self.IRIS_MIN_RESOLUTION}+)'
                }
            
            # 2. Detect circular structures (pupil/iris)
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50, 
                param2=30, 
                minRadius=20, 
                maxRadius=100
            )
            
            if circles is None or len(circles[0]) == 0:
                return {
                    'is_live': False,
                    'confidence': 0.3,
                    'reason': 'No iris/pupil detected'
                }
            
            # 3. Edge quality check (printed images have poor edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < self.IRIS_EDGE_THRESHOLD:
                return {
                    'is_live': False,
                    'confidence': 0.5,
                    'reason': f'Poor edge quality (density: {edge_density:.3f})'
                }
            
            confidence = min(edge_density / self.IRIS_EDGE_THRESHOLD, 1.0)
            
            logger.info(f"Iris liveness: LIVE (edge_density={edge_density:.3f}, conf={confidence:.2f})")
            
            return {
                'is_live': True,
                'confidence': float(confidence),
                'reason': 'Live iris detected'
            }
            
        except Exception as e:
            logger.error(f"Iris liveness detection failed: {e}")
            return {'is_live': False, 'confidence': 0.0, 'reason': f'Detection error: {str(e)}'}
    
    def detect_fingerprint_liveness(self, image: np.ndarray) -> Dict:
        """
        Detect if fingerprint image is authentic
        
        Methods:
        1. Edge sharpness analysis
        2. Ridge pattern quality
        3. Contrast check
        
        Args:
            image: Grayscale fingerprint image
            
        Returns:
            {
                'is_live': bool,
                'confidence': float,
                'reason': str
            }
        """
        try:
            if image is None or image.size == 0:
                return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid image'}
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 1. Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < self.FP_EDGE_THRESHOLD:
                return {
                    'is_live': False,
                    'confidence': 0.4,
                    'reason': f'Poor fingerprint quality (edge_density: {edge_density:.3f})'
                }
            
            # 2. Contrast check
            contrast = gray.std()
            if contrast < 30:  # Low contrast indicates poor quality
                return {
                    'is_live': False,
                    'confidence': 0.5,
                    'reason': f'Low contrast image (contrast: {contrast:.1f})'
                }
            
            confidence = min((edge_density / self.FP_EDGE_THRESHOLD) * 0.6 + (contrast / 80) * 0.4, 1.0)
            
            logger.info(f"Fingerprint liveness: LIVE (edge={edge_density:.3f}, contrast={contrast:.1f}, conf={confidence:.2f})")
            
            return {
                'is_live': True,
                'confidence': float(confidence),
                'reason': 'Live fingerprint detected'
            }
            
        except Exception as e:
            logger.error(f"Fingerprint liveness detection failed: {e}")
            return {'is_live': False, 'confidence': 0.0, 'reason': f'Detection error: {str(e)}'}
    
    def _calculate_lbp_variance(self, gray_image: np.ndarray) -> float:
        """
        Calculate Local Binary Pattern variance for texture analysis
        Higher variance = more texture variation = likely real face
        Lower variance = flat texture = likely photo/screen
        """
        try:
            # Simple LBP implementation
            rows, cols = gray_image.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray_image[i, j]
                    code = 0
                    code |= (gray_image[i-1, j-1] > center) << 7
                    code |= (gray_image[i-1, j] > center) << 6
                    code |= (gray_image[i-1, j+1] > center) << 5
                    code |= (gray_image[i, j+1] > center) << 4
                    code |= (gray_image[i+1, j+1] > center) << 3
                    code |= (gray_image[i+1, j] > center) << 2
                    code |= (gray_image[i+1, j-1] > center) << 1
                    code |= (gray_image[i, j-1] > center) << 0
                    lbp[i-1, j-1] = code
            
            # Calculate variance
            variance = float(np.var(lbp))
            return variance
            
        except Exception as e:
            logger.error(f"LBP calculation failed: {e}")
            return 0.0
    
    def _check_color_distribution(self, image: np.ndarray) -> float:
        """
        Check color distribution to detect screen/photo artifacts
        Real faces have more natural color distribution
        
        Returns:
            Score between 0-1
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for hue channel
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = hist.flatten() / hist.sum()
            
            # Real faces have more diverse hue distribution
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Normalize to 0-1 (max entropy for 180 bins is ~7.5)
            score = min(entropy / 7.5, 1.0)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Color distribution check failed: {e}")
            return 0.5  # Neutral score on error


# Global instance
liveness_detector = LivenessDetector()

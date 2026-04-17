"""Eye detection module using CLIP model and face detection."""

from typing import Dict, Tuple, Optional
from PIL import Image
import numpy as np
import warnings
import torch

# Suppress warnings
warnings.filterwarnings("ignore")


class EyeDetector:
    """Detect closed eyes in images using CLIP model and face detection."""

    def __init__(self, threshold: float = 0.02, device=None):
        """
        Initialize eye detector.

        Args:
            threshold: Difference threshold for closed vs open eyes (default: 0.02)
            device: torch device to use (CPU, CUDA, or MPS)
        """
        self.threshold = threshold
        self.device = device or torch.device("cpu")

        # Lazy-loaded models
        self.clip_model = None
        self.opencv_net = None
        self.text_embeddings = None
        self.face_recognition_available = False

        # Text prompts for eye detection
        self.text_prompts = [
            "A person with closed eyes",
            "A person with eyes wide open",
            "A face with eyelids completely covering the eyes",
            "Eyes that are shut",
            "A person blinking"
        ]

    def load_models(self):
        """Lazy load CLIP model and face detection models."""
        if self.clip_model is not None:
            return  # Already loaded

        try:
            from sentence_transformers import SentenceTransformer

            # Load CLIP model
            self.clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

            # Encode text prompts once
            self.text_embeddings = self.clip_model.encode(self.text_prompts)

            print("✓ CLIP model loaded for eye detection")

        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

        # Check if face_recognition is available
        try:
            import face_recognition
            self.face_recognition_available = True
        except ImportError:
            print("⚠ face_recognition library not available (optional). Install with: pip install face-recognition")
            self.face_recognition_available = False

    def _load_opencv_face_detector(self) -> bool:
        """
        Load OpenCV DNN face detector.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self.opencv_net is not None:
            return True

        try:
            import cv2

            self.opencv_net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )

            # Set preferable backend
            self.opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            return True

        except Exception:
            return False

    def detect_faces_opencv(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using OpenCV DNN SSD.

        Args:
            image: PIL Image

        Returns:
            (startX, startY, endX, endY) or None if no face detected
        """
        if not self._load_opencv_face_detector():
            return None

        try:
            import cv2

            # Resize if too large
            max_size = 800
            if max(image.size) > max_size:
                resize_ratio = max_size / max(image.size)
                new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
                image = image.resize(new_size, Image.LANCZOS)

            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Prepare image for detection
            blob = cv2.dnn.blobFromImage(
                image_cv,
                scalefactor=1.0,
                size=(300, 300),
                mean=[104, 117, 123],
                swapRB=False,
                crop=False
            )
            self.opencv_net.setInput(blob)

            # Run face detection
            detections = self.opencv_net.forward()

            # Process detections
            if len(detections) == 0 or detections.shape[2] == 0:
                return None

            # Get the detection with highest confidence
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                return None

            # Extract face coordinates
            box = detections[0, 0, i, 3:7] * np.array([
                image_cv.shape[1], image_cv.shape[0],
                image_cv.shape[1], image_cv.shape[0]
            ])
            (startX, startY, endX, endY) = box.astype("int")

            # Add padding
            padding = 20
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(image.width, endX + padding)
            endY = min(image.height, endY + padding)

            return (startX, startY, endX, endY)

        except Exception:
            return None

    def detect_faces_fallback(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using face_recognition library (HOG).

        Args:
            image: PIL Image

        Returns:
            (left, top, right, bottom) or None if no face detected
        """
        if not self.face_recognition_available:
            return None

        try:
            import face_recognition

            # Resize if too large
            max_size = 800
            if max(image.size) > max_size:
                resize_ratio = max_size / max(image.size)
                new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
                image = image.resize(new_size, Image.LANCZOS)

            # Convert to numpy array
            image_array = np.array(image)

            # Detect faces
            face_locations = face_recognition.face_locations(
                image_array,
                number_of_times_to_upsample=0,
                model="hog"
            )

            if not face_locations:
                return None

            # Get the first face found
            top, right, bottom, left = face_locations[0]

            # Add padding
            padding = 10
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(image.height, bottom + padding)
            right = min(image.width, right + padding)

            return (left, top, right, bottom)

        except Exception:
            return None

    def crop_to_face(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """
        Crop image to face region using three-tier fallback.

        Args:
            image: PIL Image

        Returns:
            (cropped_image, method) where method is one of:
            - 'opencv_dnn': OpenCV DNN SSD detection
            - 'face_recognition': face_recognition library
            - 'full_image': No face detected, using full image
        """
        # Try OpenCV DNN first
        face_coords = self.detect_faces_opencv(image)
        if face_coords is not None:
            startX, startY, endX, endY = face_coords
            cropped = image.crop((startX, startY, endX, endY))
            return (cropped, 'opencv_dnn')

        # Fallback to face_recognition
        face_coords = self.detect_faces_fallback(image)
        if face_coords is not None:
            left, top, right, bottom = face_coords
            cropped = image.crop((left, top, right, bottom))
            return (cropped, 'face_recognition')

        # Final fallback: use full image
        return (image, 'full_image')

    def detect_eyes(self, image: Image.Image) -> Dict:
        """
        Detect eye status in image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with:
            - status: 'open', 'closed', 'no_face', or 'error'
            - score: closed_score - open_score (positive = closed, negative = open)
            - confidence: maximum similarity score
            - method: detection method used ('opencv_dnn', 'face_recognition', 'full_image')
        """
        # Ensure models are loaded
        if self.clip_model is None:
            self.load_models()

        try:
            # Crop to face region
            face_image, method = self.crop_to_face(image)

            # Encode the cropped image
            img_embedding = self.clip_model.encode(face_image)

            # Compute similarities with all text prompts
            similarity_scores = self.clip_model.similarity(img_embedding, self.text_embeddings)

            # Extract scores for closed eyes (index 0) and open eyes (index 1)
            closed_eyes_score = similarity_scores[0][0].item()
            open_eyes_score = similarity_scores[0][1].item()

            # Calculate difference score
            score = closed_eyes_score - open_eyes_score

            # Determine status
            if method == 'full_image':
                # Full image means no face was detected
                status = 'no_face' if score > self.threshold else 'open'
            else:
                status = 'closed' if score > self.threshold else 'open'

            # Get confidence (max similarity across all prompts)
            confidence = float(similarity_scores[0].max().item())

            return {
                'status': status,
                'score': float(score),
                'confidence': confidence,
                'method': method
            }

        except Exception as e:
            return {
                'status': 'error',
                'score': 0.0,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }

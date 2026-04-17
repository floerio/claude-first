from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import face_recognition
import cv2
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load CLIP model
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

# Define the folder path
folder_path = "/Users/ofloericke/images"

def crop_to_face(image_path):
    """Crop image to face region using OpenCV face detection"""
    try:
        # Load image
        pil_image = Image.open(image_path)
        
        # Resize if too large
        max_size = 800
        if max(pil_image.size) > max_size:
            resize_ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * resize_ratio), int(pil_image.height * resize_ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Load OpenCV's pre-trained face detection model
        try:
            net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
            
            # Set preferable backend
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        except Exception:
            return crop_to_face_fallback(image_path)
        
        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(
            image_cv, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=[104, 117, 123],
            swapRB=False,
            crop=False
        )
        net.setInput(blob)
        
        # Run face detection
        detections = net.forward()
        
        # Process detections
        if len(detections) == 0 or detections.shape[2] == 0:
            return pil_image
        
        # Get the detection with highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        if confidence < 0.5:
            return pil_image
        
        # Extract face coordinates
        box = detections[0, 0, i, 3:7] * np.array([image_cv.shape[1], image_cv.shape[0],
                                                   image_cv.shape[1], image_cv.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Add padding
        padding = 20
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(image_cv.shape[1], endX + padding)
        endY = min(image_cv.shape[0], endY + padding)
        
        # Crop and return
        cropped = pil_image.crop((startX, startY, endX, endY))
        return cropped
        
    except Exception:
        return crop_to_face_fallback(image_path)

def crop_to_face_fallback(image_path):
    """Fallback to face_recognition library"""
    try:
        # Load image
        pil_image = Image.open(image_path)
        
        # Resize if too large
        max_size = 800
        if max(pil_image.size) > max_size:
            resize_ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * resize_ratio), int(pil_image.height * resize_ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
        # Convert to numpy array
        image = face_recognition.load_image_file(image_path)
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=0,
            model="hog"
        )
        
        if not face_locations:
            return pil_image
        
        # Get the first face found
        top, right, bottom, left = face_locations[0]
        
        # Add padding
        padding = 10
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(image.shape[0], bottom + padding)
        right = min(image.shape[1], right + padding)
        
        # Crop and return
        cropped = pil_image.crop((left, top, right, bottom))
        return cropped
        
    except Exception:
        return Image.open(image_path)

# Text prompts for eye detection
text_prompts = [
    "A person with closed eyes",
    "A person with eyes wide open",
    "A face with eyelids completely covering the eyes",
    "Eyes that are shut",
    "A person blinking"
]

# Encode all text prompts
text_embeddings = model.encode(text_prompts)

if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            
            try:
                # Crop to face
                face_image = crop_to_face(image_path)
                
                # Encode the cropped image
                img_emb = model.encode(face_image)
                
                # Compute similarities
                similarity_scores = model.similarity(img_emb, text_embeddings)
                
                # Extract scores
                closed_eyes_score = similarity_scores[0][0].item()
                open_eyes_score = similarity_scores[0][1].item()
                
                # Determine if eyes are closed
                is_closed_val = closed_eyes_score - open_eyes_score
                is_closed = is_closed_val > 0.02
                prediction = "closed" if is_closed else "open"
                
                print(f"{filename}: Eyes likely {prediction} val: {is_closed_val}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
import numpy as np
from pydantic import BaseModel
import cv2
from io import BytesIO

class TextObject:
    def __init__(self, text, confidence, bbox):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.isSensitive = False
        self.objectType = ""

    def __str__(self):
        return f"Text: {self.text}, Confidence: {self.confidence}, BBox: {self.bbox}"
    
    def to_dict(self):
        # Convert numpy types to Python native types
        return { 
            "text": str(self.text),
            "confidence": float(self.confidence),
            "bbox": [[float(x) for x in point] for point in self.bbox],
            "isSensitive": bool(self.isSensitive),
            "objectType": str(self.objectType)
        }

class Policy(BaseModel):
    name: str
    action: str
    pattern: str
    engines: list[str]


    
def detect_orientation(image_data: BytesIO, threshold=5):
    # Get the bytes from BytesIO
    image_bytes = image_data.getvalue()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours found!")
        return 0
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle that encloses the largest contour
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    
    # Adjust the angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # If the angle is within the threshold, return 0
    if abs(angle) < threshold:
        return 0
    
    return angle

def rotate_image(image_data: BytesIO, angle):
    # Get the bytes from BytesIO
    image_bytes = image_data.getvalue()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Get the dimensions
    (h, w) = img.shape[:2]
    
    # Calculate the center
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation
    rotated = cv2.warpAffine(
        img, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    # Convert to grayscale for finding non-white area
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    
    # Find non-white pixels
    coords = cv2.findNonZero(cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1])
    
    if coords is not None:
        # Get the bounding box of non-white pixels
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image to the content
        rotated = rotated[y:y+h, x:x+w]
    
    # Encode the rotated and cropped image back to bytes
    is_success, buffer = cv2.imencode('.jpg', rotated, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not is_success:
        raise ValueError("Failed to encode rotated image")
    
    return BytesIO(buffer.tobytes())
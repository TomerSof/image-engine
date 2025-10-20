# easy_ocr.py
import easyocr
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
from utility import TextObject




def censor_image(image_data: BytesIO, sensitiveObjs, blur_strength=51):
    # Initialize EasyOCR reader with English, Spanish, and French
    reader = easyocr.Reader(['en', 'es', 'fr'])     
    # Convert BytesIO to numpy array
    image = Image.open(image_data)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for obj in sensitiveObjs:
        # Extract coordinates
        top_left = (obj.bbox[0][0], obj.bbox[0][1])
        bottom_right = (obj.bbox[2][0], obj.bbox[2][1])

        # Extract the region of interest (ROI)
        roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Apply Gaussian Blur to the ROI
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        # Replace the original ROI with the blurred ROI
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi

    # Convert back to BytesIO
    is_success, buffer = cv2.imencode('.jpeg', img)
    if not is_success:
        raise Exception("Failed to encode image")
    
    censored_image = BytesIO(buffer.tobytes())
    return censored_image

def process_image(image_data: BytesIO):
    # Initialize EasyOCR reader with English, Spanish, and French
    reader = easyocr.Reader(['en', 'es', 'fr'])
    try:
        # Convert BytesIO to numpy array
        image = Image.open(image_data)
        image_np = np.array(image)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply Unsharp Masking for better text detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.5)
        unsharp_masked = cv2.addWeighted(gray, 2, blurred, -1, 10)

        # Perform OCR
        results = reader.readtext(unsharp_masked)
        
        # Create TextObjects with confidence threshold
        confidence_threshold = 0.01
        text_objects = []
        for bbox, text, prob in results:
            if prob >= confidence_threshold:
                new_object = TextObject(text, float(prob), bbox)
                text_objects.append(new_object)
        
        return text_objects
        
    except Exception as e:
        return {"error": f"An error occurred during OCR processing: {str(e)}"}


def process_image_by_path(image_path):
    # Initialize EasyOCR reader with English, Spanish, and French
    reader = easyocr.Reader(['en', 'es', 'fr'])
    
    # Load the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Unsharp Masking
    blurred = cv2.GaussianBlur(gray, (3, 3), 1.5)
    unsharp_masked = cv2.addWeighted(gray, 2, blurred, -1, 10)

    # Perform OCR on the unsharp masked image
    result = reader.readtext(unsharp_masked)

    # Collect recognized text above a confidence threshold
    confidence_threshold = 0.33
    recognized_text = []

    for (bbox, text, prob) in result:
        if prob >= confidence_threshold:
            recognized_text.append((text, prob))

    return recognized_text

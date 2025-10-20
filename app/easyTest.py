import easyocr
from io import BytesIO
import cv2
import numpy as np
from utility import TextObject

def is_noisy_by_laplacian(image, threshold=200):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold

def estimate_noise_std(image, patch_size=10):
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(np.std(patch))
    return np.mean(patches)

def is_noisy_std(image, threshold=12):
    std = estimate_noise_std(image)
    return std > threshold

def is_image_noisy(image):
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    std = estimate_noise_std(image)
    return lap_var > 200 and std > 12


def process_image(image_data: BytesIO):
    # Convert image bytes to numpy array
    image_bytes = image_data.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], recognizer='Transformer')  # Add other languages if needed

    # Convert image to grayscale for better OCR results
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image (adjust according to text size)
    resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(resized_image)

    blurred = cv2.GaussianBlur(enhanced_image, (1, 1), 0)
    
    _, otsu_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

     # Dilation to recover thin characters
    kernel = np.ones((1, 1), np.uint8)
    dilated_otsu = cv2.dilate(otsu_image, kernel, iterations=1)
    clean_otsu = cv2.morphologyEx(dilated_otsu, cv2.MORPH_OPEN, kernel)

    # Apply adaptive thresholding (binarize the image)
    adaptive_thresh_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

    hybrid = cv2.bitwise_and(otsu_image, adaptive_thresh_image)

    cv2.imshow('resized_image', resized_image)
    cv2.imshow('enhanced_image', enhanced_image)
    cv2.imshow('clean_otsu', clean_otsu)
    cv2.imshow('adaptive_thresh_image', adaptive_thresh_image)
    cv2.imshow('dilated_otsu', dilated_otsu)
    cv2.imshow('hybrid', hybrid)
    cv2.waitKey(0)
    
    # Compare both OCR results
    results_otsu = reader.readtext(hybrid)
    results_clean_otsu = reader.readtext(clean_otsu)
    results_adaptive = reader.readtext(adaptive_thresh_image)

    mean_otsu = np.mean([r[2] for r in results_otsu]) if results_otsu else 0
    mean_clean_otsu = np.mean([r[2] for r in results_clean_otsu]) if results_clean_otsu else 0
    mean_adaptive = np.mean([r[2] for r in results_adaptive]) if results_adaptive else 0

    results = results_otsu if mean_otsu >= mean_adaptive else results_adaptive

    print(f'mean_otsu: {mean_otsu},mean_clean_otsu: {mean_clean_otsu} mean_adaptive {mean_adaptive}')

    # Convert results to structured output
    text_objects = []
    for bbox, text, prob in results:
            text_objects.append(TextObject(text, float(prob), bbox))
    
    return text_objects

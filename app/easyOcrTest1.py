import easyocr
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
from utility import TextObject
from PIL import ImageEnhance, ImageOps

def analyze_image(image_data: BytesIO):
    """
    Analyze the image to determine appropriate preprocessing parameters
    based on its characteristics.
    Returns a dictionary with suggested values for each preprocessing step.
    """
    image = Image.open(image_data)
    image_np = np.array(ImageOps.grayscale(image))  # Convert to grayscale for analysis

    # Step 1: Calculate brightness and contrast metrics
    mean_brightness = np.mean(image_np)  # Average pixel intensity
    contrast = np.max(image_np) - np.min(image_np)  # Contrast as intensity range

    # Step 2: Determine noise level (variance of pixel intensities)
    noise_level = np.var(image_np)

    # Step 3: Set parameters based on analysis
    parameters = {
        "contrast_factor": 6 if mean_brightness < 120 else 4,  # Higher for darker images
        "resize_factor": 3.0 if noise_level < 400 else 2.5,  # Larger for clean text
        "denoising_strength": 25 if noise_level > 800 else 15,  # Stronger for noisier images
        "thresholding_method": "adaptive" if contrast < 60 else "otsu",  # Adaptive for low contrast
        "apply_edge_detection": contrast > 70 and noise_level > 600,  # Use edges for highly noisy and high-contrast images
        "morphological_kernel_size": (3, 3) if mean_brightness < 100 else (2, 2)  # Adjust kernel size dynamically
    }

    return parameters

def process_image(image_data: BytesIO):
    reader = easyocr.Reader(['en'], recognizer='Transformer')  # Add/remove languages as needed
    try:
        # Analyze the image to determine preprocessing parameters
        parameters = analyze_image(image_data)

        image = Image.open(image_data)

        # Step 1: Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(parameters["contrast_factor"])

        # Step 2: Convert to grayscale
        image = ImageOps.grayscale(image)
        image_np = np.array(image)

        # Step 3: Resize image to improve text clarity
        resized_image = cv2.resize(
            image_np, None, fx=parameters["resize_factor"], fy=parameters["resize_factor"], interpolation=cv2.INTER_LINEAR
        )

        # Step 4: Apply noise reduction
        denoised_image = cv2.fastNlMeansDenoising(resized_image, None, h=parameters["denoising_strength"])

        # Step 5: Apply thresholding dynamically
        if parameters["thresholding_method"] == "adaptive":
            processed_image = cv2.adaptiveThreshold(
                denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:  # Otsu thresholding
            _, processed_image = cv2.threshold(
                denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Step 6: Apply edge detection (optional)
        if parameters["apply_edge_detection"]:
            processed_image = cv2.Canny(processed_image, threshold1=50, threshold2=150)

        # Step 7: Apply morphological operations to refine edges
        kernel = np.ones(parameters["morphological_kernel_size"], np.uint8)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

        # Display the processed image
        processed_pil_image = Image.fromarray(processed_image)
        processed_pil_image.show()

        # Perform OCR
        results = reader.readtext(processed_image)

        # Collect TextObjects
        confidence_threshold = 0.01  # Keep low for testing; increase later
        text_objects = []
        for bbox, text, prob in results:
            text_objects.append(TextObject(text, float(prob), bbox))

        return text_objects

    except Exception as e:
        return {"error": f"An error occurred during OCR processing: {str(e)}"}
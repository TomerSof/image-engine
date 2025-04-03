from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from io import BytesIO
import json
import easy
import Mistral
from utility import Policy, detect_orientation, rotate_image
import base64

app = FastAPI()

@app.get("/")
async def root():
    # Verify DB connection
    return {"message": "Hello World", "status": "OK"}

@app.post("/detect-text/")
async def detect_text(policy: str = Form(...), file: UploadFile = File(...)):
    print("Received request")
    try:
        print("Starting detection...")
        # Parse the policy JSON string into Policy object
        policy_dict = json.loads(policy)
        policy_obj = Policy(**policy_dict)

        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG file.")

        # Read file contents
        contents = await file.read()
        byte_data = BytesIO(contents)

        print("Checking orientation...")
        # Check orientation and rotate if needed
        angle = detect_orientation(byte_data)
        if angle != 0:
            byte_data = rotate_image(byte_data, angle)

        print("Processing image...")
        # Get text objects from EasyOCR
        text_objects = easy.process_image(byte_data)
        
        # If no text was detected, return early
        if isinstance(text_objects, dict) and "error" in text_objects:
            raise HTTPException(status_code=400, detail=text_objects["error"])
        
        print("Classifying text objects...")
        # Classify text objects using Mistral
        classified_objects = Mistral.classify_text_objects(text_objects, policy_obj)
        
        # Convert list of TextObjects to list of dictionaries for JSON response
        response_data = [obj.to_dict() for obj in classified_objects]
        
        print("Creating censored image...")
        # Create censored image
        byte_data.seek(0)  # Reset buffer position
        censored_image = easy.censor_image(byte_data, [obj for obj in classified_objects if obj.isSensitive == True])
        if angle != 0:
            censored_image = rotate_image(censored_image, -angle)

        # Convert censored image to base64
        image_base64 = base64.b64encode(censored_image.getvalue()).decode('utf-8')
        
        print("Returning response...")
        return {
            "text_detection_results": response_data,
            "censored_image": {
                "filename": f"censored_{file.filename}",
                "content_type": file.content_type,
                "data": f"data:{file.content_type};base64,{image_base64}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/test")
async def test():
    print("Test endpoint hit!")
    return {"message": "API is working"}
  
  
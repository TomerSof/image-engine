from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import json
from io import BytesIO
import easyTestingMan as easy
from utility import Policy, detect_orientation, rotate_image
from ollama import Client

client = Client(host='http://host.docker.internal:11434')
app = FastAPI()


@app.post("/detect-test/")
async def detect_text(policy: str = Form(...), file: UploadFile = File(...)):

    try: # Path to the image file
        print("Starting detection...")
        # Parse the policy JSON string into Policy objqect
        policy_dict = json.loads(policy)
        policy_obj = Policy(**policy_dict)

        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG file.") 
        
        # Read file contents
        contents = await file.read()
        byte_data = BytesIO(contents)


        # received_file_path = 'cc.jpeg'  # Replace with the path to your image

        print("Checking orientation...")
        # Check orientation and rotate if needed
        angle = detect_orientation(byte_data)
        if angle != 0:
            byte_data = rotate_image(byte_data, angle)

        print("Processing image...")
        # Get text objects from EasyOCR
        text_objects = easy.process_image(byte_data)

        print("Processing text objects...")
        # Build prompt for LLM from TextObject list
        text_blocks = [f"{i+1}. {obj.text}" for i, obj in enumerate(text_objects)]
        prompt = (
            "These are text blocks extracted from an image:\n\n" +
            "\n".join(text_blocks) +
            "\n\nAssign a descriptive title to each block and fix any OCR errors if present. "
            "Please respond in this format:\nBlock #: [Corrected Text] - Title"
        )

        print("Querying LLM...")
        # Query the local LLM via Ollama
        try:
            response = client.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": "You are an OCR correction assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            print("Raw LLM response:", response)
            llm_content = response['message']['content']
        except Exception as e:
            print("LLM call failed:", e)
            raise

        print("LLM Response:\n", response['message']['content'])
        llm_content = response['message']['content']

        # Prepare JSON response
        extracted_text = [{
            "text": obj.text,
            "confidence": obj.confidence,
            "bbox": obj.bbox,
            "height": obj.height,
            "bestScale": obj.bestScale
        } for obj in text_objects]

        

        # Step 3: Print the results
        if isinstance(text_objects, list):  # Ensure it's a list
            print("Extracted text objects:")
            for obj in text_objects:
                # Assuming each `obj` has 'text', 'confidence', and 'bbox'
                print(f"Text: {obj.text}, confidence: {obj.confidence:.2f}, bbox: {obj.bbox}, height: {obj.height}, scale: {obj.bestScale}")
        else:
            print("No text objects found or an error occurred.")

        return {
            "text_objects": extracted_text,
            "llm_response": llm_content
        }
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.get("/test")
async def test():
    return {"message": "API is working"}
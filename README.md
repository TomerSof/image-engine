# **<u>image-engine**</u>

### Using postman:
* Create a request to http://127.0.0.1:8000/detect-text/
* Edit body to use "form-data" with following variables
    *  **<u>Key**</u> = "policy" --> **<u>Value**</u> = "{"name": "test", "action": "detect", "pattern": "credit card number", "engines": ["image-engine"]}"
    *  **<u>Key**</u> = "file" --> change to **<u>File**</u> option --> choose one of the CC images
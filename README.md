# image-engine

## Using postman:
* Create a request to http://127.0.0.1:8000/detect-text/
* Edit body to use "form-data" with following variables
    *  Key = "policy" --> Value = "{"name": "test", "action": "detect", "pattern": "credit card number", "engines": ["image-engine"]}"
    *  Key = "file" --> change to File option --> choose one of the CC images
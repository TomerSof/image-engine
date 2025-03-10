# **<u>image-engine</u>**

### Using Postman
To create a request to `http://127.0.0.1:8000/detect-text/`, set the body to "form-data" and include the following variables:

| **Key**      | **Type** | **Value**                                                                                            |
|--------------|----------|------------------------------------------------------------------------------------------------------|
| **policy**   | Text     | `{"name": "test", "action": "detect", "pattern": "credit card number", "engines": ["image-engine"]}` |
| **file**     | File     | CC.jpeg / CC2.jpeg                                                                                  |

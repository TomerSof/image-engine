# **<u>image-engine</u>**

### Using postman:
* Create a request to http://127.0.0.1:8000/detect-text/
* Edit body to use "form-data" with the following variables:
|   **Key**     |   **Type**    | **Value**                                                                                            |
|---------------|---------------|------------------------------------------------------------------------------------------------------|
| **policy**    |   Text        | `{"name": "test", "action": "detect", "pattern": "credit card number", "engines": ["image-engine"]}` |
| **file**      | File          | CC.jpeg / CC2.jpeg                                                                                   |


 #You need to write a script that uses the request module to make a POST request to your deployed application. 
 #Be sure to retrieve both the status code and model inference result and 
 #include this as a screenshot named live_post.png in your re-submission

import requests
import json
data = json.dumps({
    "age": 40,
    "workclass": "Private",
    "fnlgt": 193524,
    "education": "Doctorate",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States"
    })
post_url = 'https://udc-salary-predictor1.herokuapp.com/forecasts'
print(f'post request data: {data}')
print(f'post url: {post_url}')
response = requests.post(post_url, data=data)

print(f'response.status_code: {response.status_code}')
print(f'response.json(): {response.json()}')
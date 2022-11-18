from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from ..main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get():
    r = client.get("/forecasts/1")
    assert r.status_code == 200 and r.json() == {"fetch": f"Cheers from item 1"}

#You should write at least three test cases -
#A test case for the GET method. This MUST test both the status code as well as the contents of the request object.
#One test case for EACH of the possible inferences (results/outputs) of the ML model.

def test_post_salary_under_50k():
    data = json.dumps({
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        })
    r = client.post("/forecasts/", data=data)
    assert r.status_code == 200 and r.json() == {"salary_over_50k": "False"}


def test_post_salary_over_50k():
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
    r = client.post("/forecasts/", data=data)
    assert r.status_code == 200 and r.json() == {"salary_over_50k": "True"}
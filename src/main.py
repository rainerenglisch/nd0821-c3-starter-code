# Put the code for your API here.
# uvicorn  starter.main:app --reload 
import os
from joblib import  load
import pandas as pd
import numpy as np

from src.starter.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -d myremote -f s3://udc-ml-devops-project3")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI, Body

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class InputFeatures(BaseModel):
    age: int
    workclass: str
    fnlgt:int
    education:str 
    #education_num:int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain : int
    capital_loss : int 
    hours_per_week: int
    native_country: str 
    #salary: bool

app = FastAPI()

#read model
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

fname_model = './src/model/model.joblib'
fname_encoder = './src/model/cat_encoder.joblib'
#fname_feature_names = './starter/model/feature_names.txt'
#model=pickle.load(open(fname_model, 'rb'))
#encoder=pickle.load(open(fname_encoder, 'rb'))
with open(fname_model, 'rb') as f:
    model=load(f)
with open(fname_encoder, 'rb') as f:
    encoder=load(f)


@app.get("/forecasts/{item_id}")
async def get_items(item_id: int):
    return {"fetch": f"Cheers from item {item_id}"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/forecasts/")
async def forecast(item: InputFeatures = Body(
    example={
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
        }
)):
    # convert to pandas dataframe
    data = [item]
    X = pd.DataFrame([s.__dict__ for s in data])
    # separate continous features
    X_continuous = X.drop(*[cat_features], axis=1)
    # one hot encode categorical features
    X_categorical = encoder.transform(X[cat_features])
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    X= pd.DataFrame(X, columns=list(X_continuous.columns.values) + list(encoder.get_feature_names_out()))
    preds = inference(model, X)
    print(preds[0])
    return {"salary_over_50k": str(preds[0]==1)}

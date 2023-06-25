#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

model = joblib.load('house_prediction_model-no_drop_features.pkl')                                                                        #define inputclass

app = FastAPI()

class MyInput(BaseModel):
    MSZoning: str
    LotShape: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    YearRemodAdd: int
    RoofStyle: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    FirstFlrSF: int
    GrLivArea: int
    BsmtFullBath: int
    KitchenQual: str
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageFinish: str
    GarageCars: int
    GarageArea: int
    GarageCond: str
    PavedDrive: str
    SaleCondition: str

@app.post('/predict/')
async def predict(input: MyInput):
    data = input.dict()
    data_ = [data['MSZoning'], data['LotShape'], data['Neighborhood'], data['Condition1'], data['Condition2'], 
              data['BldgType'], data['HouseStyle'], data['OverallQual'], data['YearRemodAdd'], data['RoofStyle'],
              data['MasVnrType'], data['MasVnrArea'], data['ExterQual'], data['BsmtQual'], data['BsmtCond'],
              data['BsmtExposure'], data['BsmtFinType1'], data['FirstFlrSF'], data['GrLivArea'], data['BsmtFullBath'],
              data['KitchenQual'], data['Functional'], data['Fireplaces'], data['FireplaceQu'], data['GarageType'],
              data['GarageFinish'], data['GarageCars'], data['GarageArea'], data['GarageCond'], data['PavedDrive'], 
              data['SaleCondition']]

    # Converting data_ to pandas DataFrame
    df = pd.DataFrame([data_], columns=data.keys())

    prediction = model.predict(df)[0]

    return {
        'prediction': prediction
    }


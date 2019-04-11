import pandas as pd
import numpy as np
from scripts.utils import load_dataset
from sklearn.externals import joblib
from scripts.utils import load_dataset

dtypes = dict(
    id=int,
    age=float,
    monthly_income=float,
    collateral_value=float,
    loan_amount=float,
    city=str,
    state=str,
    collateral_debt=float,
    verified_restriction="category",
    dishonored_checks="category",
    expired_debts="category",
    banking_debts="category",
    commercial_debts="category",
    protests="category",
    marital_status=str,
    informed_restriction="category",
    loan_term=float,
    monthly_payment=float,
    informed_purpose=str,
    auto_brand=str,
    auto_model=str,
    auto_year=float,
    pre_approved="category",
    form_completed="category",
    channel=str,
    zip_code=str,
    landing_page=str,
    landing_page_product=str,
    gender=str,
    utm_term=str,
    education_level=str
)

dataset = load_dataset(dataset="input.csv", dtypes=dtypes, path="input/")
dataset.drop('informed_purpose', axis=1, inplace=True)

missing = joblib.load("model_files/exclude_missing.pkl")
dataset.drop(np.array(missing), axis=1, inplace=True)

scaler = joblib.load("model_files/scaler.pkl")

int_cols = list(dataset.select_dtypes(include=['category']))
dataset[int_cols] = dataset[int_cols].replace(np.nan, -1)
dataset[int_cols] = dataset[int_cols].astype("float64").astype("int64")

obj_cols = list(dataset.select_dtypes(include=['object']))
dataset[obj_cols] = dataset[obj_cols].replace(np.nan, "missing").astype("category")

num_cols = list(dataset.select_dtypes(include=['float64']))

for c in num_cols:
    value = joblib.load("model_files/" + c + "_fill_na_value.pkl")
    dataset[c] = dataset[c].fillna(value)

cat_cols = list(dataset.select_dtypes(include=['category']))
num_cols = list(dataset.select_dtypes(include=['float64']))

dataset[num_cols] = scaler.transform(dataset[num_cols])

for i, c in enumerate(cat_cols):
    filename = "model_files/" + c + "_mlb.pkl"
    multilabel = joblib.load(filename)
    
    dataset = dataset.merge(multilabel, how='left', left_on=c, right_index=True)
    dataset.drop(c, axis=1, inplace=True)

# Fix multilabels after left join
dataset.fillna(0, inplace=True)

dataset.to_pickle("model_files/preprocessed_input.pkl")

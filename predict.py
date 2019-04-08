import pandas as pd
import numpy as np
from scripts.utils import load_dataset
from sklearn.externals import joblib

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
    informed_restriction=float,
    loan_term=float,
    monthly_payment=float,
    informed_purpose=str,
    auto_brand=str,
    auto_model=str,
    auto_year=float,
    pre_approved="category",
    form_completed=float,
    sent_to_analysis=float,
    channel=str,
    zip_code=str,
    landing_page=str,
    landing_page_product=str,
    gender=str,
    utm_term=str,
    education_level=str
)

data = load_dataset(dataset="input.csv", dtypes=dtypes, path="input/")


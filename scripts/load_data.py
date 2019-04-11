from utils import load_dataset, save_pickle
import os

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
    sent_to_analysis=float,
    channel=str,
    zip_code=str,
    landing_page=str,
    landing_page_product=str,
    gender=str,
    utm_term=str,
    education_level=str
)

raw_data = load_dataset(dataset="dataset.csv", dtypes=dtypes, path="data/")

if not os.path.exists("model_files/"):
    os.makedirs("model_files/")

raw_data.to_pickle("model_files/raw_data.pkl")

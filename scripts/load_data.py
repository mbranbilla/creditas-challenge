from scripts.utils import load_dataset, save_pickle

dtypes = dict(
    id=int,
    age=float,
    monthly_income=float,
    collateral_value=float,
    loan_amount=float,
    city=str,
    state=str,
    collateral_debt=float,
    verified_restriction=float,
    dishonored_checks=int,
    expired_debts=int,
    banking_debts=int,
    commercial_debts=int,
    protests=int,
    marital_status=str,
    informed_restriction=float,
    loan_term=float,
    monthly_payment=float,
    informed_purpose=str,
    auto_brand=str,
    auto_model=str,
    auto_year=float,
    pre_approved=float,
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

raw_data = load_dataset(dataset="dataset.csv", dtypes=dtypes, path="data/")

save_pickle(raw_data, "raw_data.pkl", "model_files/")

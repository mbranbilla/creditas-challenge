import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from scripts.utils import filter_outliers

# Read dataset from pickle with dtypes
dataset = pd.read_pickle("model_files/raw_data.pkl")

# Remove data with missing target sent_to_analysis
dataset = dataset[~dataset['sent_to_analysis'].isna()]

# Exclude columns that have high occurrences of missing values (missing_rate >= .25)
missing = pd.DataFrame(1 - (dataset.count() / dataset['id'].count()), columns=['missing_rate'])

dataset.drop(
    missing[missing['missing_rate'] >= .25].index,
    axis=1,
    inplace=True
)

joblib.dump(missing, "model_files/exclude_missing.pkl")

# Drop id (it isn`t a predictor) and "informed_purpose" column
dataset.drop(["id", "informed_purpose"], axis=1, inplace=True)

# Fix dtypes of int categorical data and str categorical data 
# and fill missing values with new category
int_cols = list(dataset.select_dtypes(include=['category']))
dataset[int_cols] = dataset[int_cols].replace(np.nan, -1).astype(int)

obj_cols = list(dataset.select_dtypes(include=['object']))
dataset[obj_cols] = dataset[obj_cols].replace(np.nan, "missing").astype("category")

dataset[['informed_restriction', 'form_completed']] = dataset[
    ['informed_restriction', 'form_completed']
    ].fillna(-1)

dataset[['informed_restriction', 'form_completed', 'sent_to_analysis']] = dataset[
    ['informed_restriction', 'form_completed', 'sent_to_analysis']
    ].astype('int64')

# Fill missing for numerical values (float)
num_cols = list(dataset.select_dtypes(include=['float64']))

for c in num_cols:
    dataset[c] = dataset[c].fillna(dataset[c].median())

# Outlier filter
num_cols = list(dataset.select_dtypes(include=['float64']))

for c in num_cols:
    dataset = filter_outliers(dataset, c, n_stds=3)

    
# Set scaler and save pickle
scaler = StandardScaler()

num_cols = list(dataset.select_dtypes(include=['float64']))

scaler.fit(dataset[num_cols])

joblib.dump(scaler, "model_files/scaler.pkl")

dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

# Binnarizer categorical features
cat_cols = list(dataset.select_dtypes(include=['category']))
pkl_names = ["model_files/" + n + "_mlb.pkl" for n in cat_cols]

for i, c in enumerate(cat_cols):
    multilabel = pd.DataFrame(MultiLabelBinarizer().fit_transform(dataset[c].astype(str).unique()))
    multilabel.columns = [c + "_" + str(n) for n in multilabel.columns]
    multilabel.set_index(dataset[c].astype(str).unique(), inplace=True)

    dataset = dataset.merge(multilabel, how='left', left_on=c, right_index=True)

    dataset.drop(c, axis=1, inplace=True)

    with open(pkl_names[i], 'wb') as f:
        joblib.dump(multilabel, f)

# Export current dataset
dataset.to_pickle("model_files/preprocessed_dataset.pkl")

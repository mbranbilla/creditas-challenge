import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib

features = pd.read_pickle("model_files/preprocessed_input.pkl")

num_cols = list(features.select_dtypes(include=['float64']))

poly = PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=False
).fit(features[num_cols])

interactions = pd.DataFrame(
    data=poly.fit_transform(
        features[num_cols]
    ),
    columns=poly.get_feature_names(num_cols)
)

interactions.drop(num_cols, axis=1, inplace=True) 

interactions.set_index(features.index, inplace=True)

features = features.merge(interactions, left_index=True, right_index=True)

low_var_to_delete = joblib.load("model_files/low_var_features.pkl")
features.drop(low_var_to_delete, axis=1, inplace=True)

features.to_pickle("model_files/input_features.pkl")

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SVMSMOTE

features = pd.read_pickle("model_files/preprocessed_dataset.pkl")

# Create relationship between numerical

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

interactions.drop(num_cols, axis=1, inplace=True)  # Remove the original columns
interactions.set_index(features.index, inplace=True)

features = features.merge(interactions, left_index=True, right_index=True)

# Remove low-variance features
not_select = ['sent_to_analysis']

selector = VarianceThreshold(threshold=.01)
selector.fit(features.drop(not_select, axis=1))

f = np.vectorize(lambda x : not x)

v = features.drop(not_select, axis=1).columns[f(selector.get_support())]

features.drop(v, axis=1, inplace=True)

# Split target from features

target = features['sent_to_analysis']
features.drop('sent_to_analysis', axis=1, inplace=True)

# Imbalanced data treatment

## 1. Using SMOTEENN
smote = SMOTEENN(random_state=0)
X_train, y_train = smote.fit_resample(features, target.values)




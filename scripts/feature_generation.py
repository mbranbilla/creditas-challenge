import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SVMSMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

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

# Remove features with low variance and save list of deleted
not_select = ['sent_to_analysis']

selector = VarianceThreshold(threshold=.01)
selector.fit(features.drop(not_select, axis=1))

f = np.vectorize(lambda x : not x)

v = features.drop(not_select, axis=1).columns[f(selector.get_support())]

features.drop(v, axis=1, inplace=True)
joblib.dump(v, "model_files/low_var_features.pkl")

# Split target from features
target = features['sent_to_analysis']
features.drop('sent_to_analysis', axis=1, inplace=True)

# Save a list with feature importance (Univariate Selection)
feature_importance = SelectKBest(k=len(features.columns)).fit(features, target)

feature_importance = pd.concat(
    [pd.DataFrame(features.columns),pd.DataFrame(feature_importance.scores_)],
    axis=1
    )
feature_importance.columns = ['feature','score']
feature_importance['score'] = feature_importance['score'].values / feature_importance['score'].sum()
feature_importance.reset_index(inplace=True, drop=True)
joblib.dump(feature_importance, "model_files/feature_importance.pkl")

# train test split and save data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

joblib.dump(pd.DataFrame(X_test, columns=feature_importance['feature']), "model_files/X_test.pkl")
y_train.to_pickle("model_files/y_train.pkl")
y_test.to_pickle("model_files/y_test.pkl")

# Imbalanced data treatment using SMOTEENN
smote = SMOTEENN(random_state=0)
X_train, y_train = smote.fit_resample(X_train, y_train)

joblib.dump(pd.DataFrame(X_train, columns=feature_importance['feature']), "model_files/X_train.pkl")
joblib.dump(y_train, "model_files/balanced_y_train.pkl")

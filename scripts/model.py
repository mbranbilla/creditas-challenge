import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV

X_train = joblib.load("model_files/X_train.pkl")
y_train = joblib.load("model_files/balanced_y_train.pkl")

model = GradientBoostingClassifier(
    n_estimators=100,
    max_features=None,
    loss='exponential',
    learning_rate=0.05
    )

model.fit(X_train, y_train)

joblib.dump(model, "model_files/model.pkl")

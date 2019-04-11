import pandas as pd
import numpy as np
from sklearn.externals import joblib
import datetime

model = joblib.load("model_files/model.pkl")
features = pd.read_pickle("model_files/input_features.pkl")

predictions = model.predict(features.drop('id', axis=1))
probabilities = model.predict_proba(features.drop('id', axis=1))

results = pd.DataFrame()
results['id'] = features['id']
results['predicted_class'] = predictions
results['class_probability'] = [p[x] for x, p in zip(results['predicted_class'], probabilities)]
results['probability_of_send_to_analysis'] = [p[0] for p in probabilities]


filename = "outputs/" + str(datetime.datetime.now()) + ".csv"
filename = filename.replace(" ", "_")
results.to_csv(filename)

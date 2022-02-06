# Script to train machine learning model.
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
from ml.data import process_data
import pickle
from ml.model import (train_model,
                      inference,
                      compute_model_metrics,
                      compute_metrics_on_slice)
import pandas as pd  # import read_csv

# Add code to load in the data.
data_path = "../data/modified_census.csv"
data = pd.read_csv(data_path)
print(data.head())
print(data.dtypes)
# Optional enhancement,
# use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary"
)

# save encoder and lb
encoder_file = '../model/encoder.sav'
lb_file = '../model/lb.sav'

pickle.dump(encoder, open(encoder_file, 'wb'))
pickle.dump(lb, open(lb_file, 'wb'))

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False,
    encoder=encoder,
    lb=lb)
# Train and save a model.
model = train_model(X_train, y_train)
filename = '../model/model.sav'
pickle.dump(model, open(filename, 'wb'))

# Inference on test data.
y_pred = inference(model, X_test)
# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("Overall Metrics")
print("""
         Prcesion: {:1.3f} \n
         Recall: {:1.3f} \n
         fbeta: {:1.3f}""".format(precision, recall, fbeta))
print()
# Compute metrics on slices
print("Model performance on slices")
for cat in cat_features:
    result = compute_metrics_on_slice(y_test, y_pred, test[cat])
    print("Model performance over {}".format(cat))
    print(result)
    print()

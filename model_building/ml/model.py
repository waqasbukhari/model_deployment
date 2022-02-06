from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RF()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_metrics_on_slice(y, preds, cat_feature):
    result = []
    # Only spit result for categories that have at least 25 data points 
    cat_count = cat_feature.value_counts()
    # retain only the categories with at least above threshold.
    categories = cat_count[cat_count>25].index
    for category in categories:
        precision, recall, fbeta = compute_model_metrics(y[cat_feature==category], preds[cat_feature==category])
        tmp_result = {"category":category, "precision":precision, "recall":recall, "fbeta":fbeta}
        result.append(tmp_result)

    return pd.DataFrame(result)
          
def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred

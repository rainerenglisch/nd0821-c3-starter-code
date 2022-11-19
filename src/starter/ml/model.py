from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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
    clf = RandomForestClassifier(random_state=0)
    model =clf.fit(X_train,y_train)
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
    preds = model.predict(X)
    return preds


def compute_model_metrics_on_slices(model, X, y, categorical_features, encoder, fname='./slice_output.txt'):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    fname : file in which metrics info is stored
    Returns
    -------
    None

    """
    #print(categorical_features)
    with open(fname, 'w') as out:
        for cat in categorical_features:
            #print(cat)
            cat_feature_names = [feat for feat in encoder.get_feature_names_out() if feat.startswith(cat)]
            #print(cat_feature_names)
            for cat_feature_name in cat_feature_names:
                #print(cat_feature_name)
                filter = X[cat_feature_name]==1.
                X_i = X.loc[filter]
                y_i = y.loc[filter]
                if X_i.size > 0:
                    preds = model.predict(X_i)
                    fbeta = fbeta_score(y_i, preds, beta=1, zero_division=1)
                    precision = precision_score(y_i, preds, zero_division=1)
                    recall = recall_score(y_i, preds, zero_division=1)
                    out.write(f'Slicing categorical feature: {cat} for value {cat_feature_name}\n')
                    out.write(f'precision={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}\n')
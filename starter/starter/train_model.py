# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
# Add code to load in the data.
data = pd.read_csv('./starter/data/census_clean.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
)

#df_describe = pd.DataFrame(y_train)
#print(df_describe.iloc[:,0].unique())
# Process the test data with the process_data function.

# Train and save a model.
print('Train model')
model = train_model(X_train, y_train)
print('Compute metrics on test set')
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'precision={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}')
filename = './starter/model/model.pickle'
print(f'Save model to {filename}')
pickle.dump(model, open(filename, 'wb'))
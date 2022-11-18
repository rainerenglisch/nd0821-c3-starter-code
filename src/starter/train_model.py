# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_on_slices
# Add code to load in the data.
data = pd.read_csv('./starter/data/census_clean.csv')
# drop eduction_num as it correlates directly to education
data=data.drop('education-num',axis='columns')
# change in column names: replace '-' with '_'
data.columns = [col.replace('-','_') for col in data.columns.values]
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

print(f'train size: {train.size}')
print(f'test size: {test.size}')

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
)

#print(f'X_train.shape: {X_train.shape}')
# save features
# with open(fname, 'w') as out:

# Process the test data with the process_data function.
print('Train model')
model = train_model(X_train, y_train)

print('Compute metrics on test set')
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'precision={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}')

print("Compute metrics for each slice of categorical feature")
compute_model_metrics_on_slices(model, X_test, y_test, cat_features, encoder)

# Train and save a model.
fname_model = './starter/model/model.pickle'
fname_encoder = './starter/model/cat_encoder.pickle'
fname_feature_names = './starter/model/feature_names.txt'
print(f'Save model to {fname_model} and encoder to {fname_encoder}')
pickle.dump(model, open(fname_model, 'wb'))
pickle.dump(encoder, open(fname_encoder, 'wb'))
with open(fname_feature_names, 'w') as out:
    out.write(str(train.columns.values))

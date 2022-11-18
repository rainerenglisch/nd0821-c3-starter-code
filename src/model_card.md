# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a trained random forest model trained on the public "Census Income Data Set":
https://archive.ics.uci.edu/ml/datasets/census+income
## Intended Use
It predicts whether income exceeds $50K/yr based on census data. 

## Training Data
The training data comprises 80% of the original data with 390720 records having 108 features. Categorical features were one hot encoded.
## Evaluation Data
Evaluation data comprises 20% of the original data with 97695 records.
## Metrics
Metrics on test set
precision=0.73, recall=0.63, fbeta=0.68

Weak performance observed at following slices. Be cautious :
Slicing categorical feature: native-country for value native-country_Thailand
precision=1.00, recall=0.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_South
precision=0.00, recall=0.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_Portugal
precision=1.00, recall=0.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_Mexico
precision=1.00, recall=0.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_Jamaica
precision=1.00, recall=0.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_Dominican-Republic
precision=0.00, recall=1.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_El-Salvador
precision=0.00, recall=1.00, fbeta=0.00
Slicing categorical feature: native-country for value native-country_France
precision=1.00, recall=0.00, fbeta=0.00
Slicing categorical feature: education for value education_1st-4th
precision=1.00, recall=0.00, fbeta=0.00
## Ethical Considerations

## Caveats and Recommendations

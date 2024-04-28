from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# List of estimators for the stacking classifier
estimators = [
    ('rf', RandomForestClassifier(max_depth=20, min_samples_leaf=10, n_estimators=61, random_state=42, n_jobs=-1)),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    ('gb', GradientBoostingClassifier(n_estimators=400, learning_rate=0.3, max_depth=10, random_state=42))
]

# Loading and displaying the first few rows of the dataset
df = pd.read_csv('Healthcare_Dataset_Preprocessed.csv')
df.head()

# Preparing feature matrix X and target vector y
X = df.drop('Target', axis=1)
y = df['Target']

# Splitting the dataset into training and temporary test sets
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.25)

# Splitting the temporary test set into validation and final test sets
X_validation, X_test, y_validation, y_test = train_test_split(X_test1, y_test1, test_size=0.50)

# Printing the shapes of the datasets
print(X_train.shape)
print(y_train.shape)
print(X_validation.shape)
print(y_validation.shape)
print(X_test.shape)
print(y_test.shape)

# Generating meta-features for training using cross-validated predictions
X_train_meta = []
for name, model in estimators:
    predictions = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
    X_train_meta.append(predictions[:, 1])
print(predictions.shape)
print(X_train_meta)
X_train_meta = np.column_stack(X_train_meta)
print(X_train_meta)

# Training the meta-model
meta_model = LogisticRegression()
meta_model.fit(X_train_meta, y_train)

def generate_meta_features(estimators, X):
    """
    Generate meta-features for a given dataset X using a list of specified estimators.

    Parameters:
    - estimators: List of tuples containing model names and their corresponding initialized objects.
    - X: Features dataset for which meta-features are to be generated.

    Returns:
    - A numpy array containing meta-features.
    """
    meta_features = []
    for name, model in estimators:
        model.fit(X_train, y_train)  # Fit each base model on the training data
        predictions = model.predict_proba(X)  # Predict probabilities
        meta_features.append(predictions[:, 1])  # Append positive class probabilities
    return np.column_stack(meta_features)

# Generating meta-features for validation and test sets
X_validation_meta = generate_meta_features(estimators, X_validation)
X_test_meta = generate_meta_features(estimators, X_test)

# Predicting and evaluating the model
y_pred_validation = meta_model.predict(X_validation_meta)
y_pred_test = meta_model.predict(X_test_meta)

# Calculating and printing F1 scores for validation and test sets
f1_score_validation = f1_score(y_validation, y_pred_validation)
print(f"f1_score_validation: {f1_score_validation}")
f1_score_test = f1_score(y_test, y_pred_test)
print(f"f1_score_test: {f1_score_test}")

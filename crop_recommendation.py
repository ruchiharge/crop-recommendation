import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

#loading and reading dataset
data_file = 'crop_recommendation.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found in the current directory.")

data = pd.read_csv(data_file)

print("\nFirst few rows of the dataset:")
print(data.head())
print("\nDataset Summary:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())

X = data.drop('label', axis=1)
y = data['label']

#splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#initializing models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

#ensembling model using soft voting
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('gbm', gbm_model)
    ],
    voting='soft'
)

#training and making predictions
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

#evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))




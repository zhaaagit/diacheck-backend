# -*- coding: utf-8 -*-
"""
Advanced Diabetes Risk Prediction Model Training
Based on the improved Colab model
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

# =============== 1) LOAD DATA ===============
print("Loading dataset...")
try:
    df = pd.read_csv("Dataset_Rill.csv", sep=';')
except FileNotFoundError:
    print("Error: 'Dataset_Rill.csv' not found.")
    exit()

df = df.drop_duplicates()
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Split data
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter grid - optimized for better accuracy
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [20, 25, 30],
    'classifier__min_samples_split': [3, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2']
}

categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

# =============== 2) CREATE QUICK MODEL (without lab tests) ===============
print("\n" + "="*60)
print("Training QUICK Model (without lab tests)...")
print("="*60)

# Features for quick model
numeric_features_quick = ['age', 'bmi']

# Preprocessing pipeline
numeric_transformer_quick = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_transformer = SimpleImputer(strategy='most_frequent')

preprocessor_quick = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_quick, numeric_features_quick),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ],
    remainder='drop'
)

# Build pipeline with SMOTE
classifier_quick = RandomForestClassifier(random_state=42, n_jobs=-1)
pipeline_quick = imbPipeline(steps=[
    ('preprocessor', preprocessor_quick),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', classifier_quick)
])

# Grid search - with more CV folds for better validation
grid_quick = GridSearchCV(pipeline_quick, param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_quick.fit(X_train, y_train)

# Evaluate
y_pred_quick = grid_quick.predict(X_test)
y_proba_quick = grid_quick.predict_proba(X_test)[:, 1]
auc_quick = roc_auc_score(y_test, y_proba_quick)
acc_quick = accuracy_score(y_test, y_pred_quick)

print(f"\n✓ Quick Model Trained!")
print(f"  Best Parameters: {grid_quick.best_params_}")
print(f"  Test AUC: {auc_quick:.4f}")
print(f"  Test Accuracy: {acc_quick:.4f}")

# Save quick model and features
joblib.dump(grid_quick.best_estimator_, 'diabetes_model_quick.pkl')
joblib.dump(numeric_features_quick, 'numeric_features_quick.pkl')
joblib.dump(categorical_features, 'categorical_features.pkl')
joblib.dump(binary_features, 'binary_features.pkl')

# =============== 3) CREATE FULL MODEL (with lab tests) ===============
print("\n" + "="*60)
print("Training FULL Model (with HbA1c & Blood Glucose)...")
print("="*60)

# Features for full model
numeric_features_full = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Preprocessing pipeline for full model
numeric_transformer_full = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_full, numeric_features_full),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ],
    remainder='drop'
)

# Build pipeline with SMOTE
classifier_full = RandomForestClassifier(random_state=42, n_jobs=-1)
pipeline_full = imbPipeline(steps=[
    ('preprocessor', preprocessor_full),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', classifier_full)
])

# Grid search
grid_full = GridSearchCV(pipeline_full, param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_full.fit(X_train, y_train)

# Evaluate
y_pred_full = grid_full.predict(X_test)
y_proba_full = grid_full.predict_proba(X_test)[:, 1]
auc_full = roc_auc_score(y_test, y_proba_full)
acc_full = accuracy_score(y_test, y_pred_full)

print(f"\n✓ Full Model Trained!")
print(f"  Best Parameters: {grid_full.best_params_}")
print(f"  Test AUC: {auc_full:.4f}")
print(f"  Test Accuracy: {acc_full:.4f}")

# Save full model and features
joblib.dump(grid_full.best_estimator_, 'diabetes_model_full.pkl')
joblib.dump(numeric_features_full, 'numeric_features_full.pkl')

print("\n✓ All files saved:")
print("  QUICK Model:")
print("    - diabetes_model_quick.pkl")
print("    - numeric_features_quick.pkl")
print("  FULL Model:")
print("    - diabetes_model_full.pkl")
print("    - numeric_features_full.pkl")
print("  Shared:")
print("    - categorical_features.pkl")
print("    - binary_features.pkl")
print("\n" + "="*60)
print("Model training complete! Ready to use in web app.")
print("="*60)

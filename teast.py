Python 3.12.4 (v3.12.4:8e8a4baf65, Jun  6 2024, 17:33:18) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
... import pandas as pd
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler
... from sklearn.linear_model import LogisticRegression
... from sklearn.metrics import accuracy_score
... from sklearn.model_selection import GridSearchCV
... 
... # Step 1: Create synthetic data
... np.random.seed(42)
... X = np.random.randn(100, 5)  # 100 samples, 5 features
... y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Target: based on first two features
... 
... # Split the data into training and test sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
... 
... # Helper function to evaluate model
... def evaluate_model(X_train, X_test, y_train, y_test, model):
...     model.fit(X_train, y_train)
...     y_pred = model.predict(X_test)
...     accuracy = accuracy_score(y_test, y_pred)
...     return accuracy
... 
... # ---- 1. Preprocessing Leakage ----
... # Apply scaling to both train and test data before split
... scaler = StandardScaler()
... X_scaled = scaler.fit_transform(X)  # Apply scaling to the entire dataset first
... 
... # Split data (after scaling, introducing leakage)
... X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
... print("Accuracy with Preprocessing Leakage:", evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, LogisticRegression()))
... 
... # ---- 2. Multi-Test Leakage ----
... # Perform grid search for hyperparameter tuning using the entire dataset, including the test set (introducing leakage)
param_grid = {'C': [0.1, 1, 10]}  # Example hyperparameter grid for Logistic Regression
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X, y)  # Fitting on the entire dataset (training + test sets)
best_model = grid_search.best_estimator_

# Now evaluate with the same test set that was part of hyperparameter tuning (introducing multi-test leakage)
y_pred = best_model.predict(X_test)
print("Accuracy with Multi-Test Leakage:", accuracy_score(y_test, y_pred))

# ---- 3. Overlap Leakage ----
# Introduce overlap leakage by copying some samples from the training set into the test set
X_train_overlap = X_train.copy()
y_train_overlap = y_train.copy()

# Add some overlap from the training set to the test set
X_test_overlap = np.vstack([X_test, X_train[:10]])  # Adding first 10 samples from train set to test set
y_test_overlap = np.hstack([y_test, y_train[:10]])

# Train the model on the training data (with the overlap) and evaluate on the test data (with the overlap)
print("Accuracy with Overlap Leakage:", evaluate_model(X_train_overlap, X_test_overlap, y_train_overlap, y_test_overlap, LogisticRegression()))

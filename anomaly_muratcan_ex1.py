import numpy as np
import time
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Reshape
X = x_train.astype("float32") / 255.0
X = x_train.reshape((X_all.shape[0], -1))
print("Normalized data:", X.shape)

# '0' is normal, others are anomalies
mask = y_train == 0
X_normals = X[mask]
X_anomalies = X[~mask]
print(f"Normal samples (label==0): {X_normals.shape[0]}")
print(f"Anomaly samples (label!=0): {X_anomalies.shape[0]}")

# Combine for training 
X_mixed = np.vstack((X_normals, X_anomalies))
y_mixed = np.hstack((np.zeros(len(X_normals)), np.ones(len(X_anomalies)))) 
print("Combined training set shape:", X_mixed.shape)
print("Label distribution:", dict(zip(*np.unique(y_mixed, return_counts=True))))

# Metric function 
def compute_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

# Isolation Forest 
print("\n Isolation Forest")
start = time.time()
iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(X_mixed)
iso_preds = np.where(iso_model.predict(X_mixed) == 1, 0, 1)
end = time.time()
iso_time = end - start
print("Prediction distribution:", dict(zip(*np.unique(iso_preds, return_counts=True))))
iso_metrics = compute_metrics(y_mixed, iso_preds)
print("Metrics:", iso_metrics)
print(f"Execution time: {iso_time:.2f} seconds")

# One-Class SVM
print("\n One-Class SVM")
start = time.time()
svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
svm_model.fit(X_mixed)
svm_preds = np.where(svm_model.predict(X_mixed) == 1, 0, 1)
end = time.time()
iso_time = end - start
svm_metrics = compute_metrics(y_mixed, svm_preds)
print("Metrics:", svm_metrics)
print(f"Execution time: {svm_time:.2f} seconds")

# LOF
print("\nLocal Outlier Factor")
start = time.time()
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_preds = np.where(lof_model.fit_predict(X_mixed) == 1, 0, 1)
end = time.time()
iso_time = end - start
lof_metrics = compute_metrics(y_mixed, lof_preds)
print("Metrics:", lof_metrics)
print(f"Execution time: {lof_time:.2f} seconds")

# Compare on test set
print("\n Test Set Evaluation")
X_test = x_test.reshape((x_test.shape[0], -1))
mask_test = y_test == 0
y_test_labels = np.where(mask_test, 0, 1)
print("Test set shape:", X_test.shape)
print("Label distribution:", dict(zip(*np.unique(y_test_labels, return_counts=True))))

# Test predictions
iso_test_preds = np.where(iso_model.predict(X_test) == 1, 0, 1)
svm_test_preds = np.where(svm_model.predict(X_test) == 1, 0, 1)
lof_test_preds = np.where(lof_model.fit_predict(X_test) == 1, 0, 1)

print("\nIsolation Forest Test Prediction Distribution:", dict(zip(*np.unique(iso_test_preds, return_counts=True))))
print("One-Class SVM Test Prediction Distribution:", dict(zip(*np.unique(svm_test_preds, return_counts=True))))
print("LOF Test Prediction Distribution:", dict(zip(*np.unique(lof_test_preds, return_counts=True))))

# Test metrics
iso_test_metrics = compute_metrics(y_test_labels, iso_test_preds)
svm_test_metrics = compute_metrics(y_test_labels, svm_test_preds)
lof_test_metrics = compute_metrics(y_test_labels, lof_test_preds)

print("\nIsolation Forest Test Metrics:", iso_test_metrics)
print("One-Class SVM Test Metrics:", svm_test_metrics)
print("Local Outlier Factor Test Metrics:", lof_test_metrics)


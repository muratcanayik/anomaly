import numpy as np
import time
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
print(f"Train set shape: {X_train_full.shape}, Test set shape: {X_test_full.shape}")

# Combine train and test sets
X_all = np.concatenate([X_train_full, X_test_full], axis=0)
y_all = np.concatenate([y_train_full, y_test_full], axis=0)
print(f"Combined dataset shape: {X_all.shape}, Labels shape: {y_all.shape}")

# Normalize
X_all = X_all.astype("float32") / 255.0
X_all = X_all.reshape((X_all.shape[0], -1))
print(f"Normalized data: {X_all.shape}")

# Use digit "0" as the normal class
X_train = X_all[y_all == 0]
print(f"Training data (only digit '0'): {X_train.shape}")

# Prepare test dataset
X_test_normal = X_all[y_all == 0][:70000]
X_test_anomaly = X_all[y_all != 0][:7000]
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.hstack([
    np.zeros(len(X_test_normal)), 
    np.ones(len(X_test_anomaly))
])
# print(f"Test normal samples: {X_test_normal.shape}")
# print(f"Test anomaly samples: {X_test_anomaly.shape}")
# print(f"Final test set shape: {X_test.shape}")
# print("Test label distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

results = []

# One-Class SVM 
print("\n One-Class SVM")
start = time.time()
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm.fit(X_train)
y_pred = ocsvm.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)
end = time.time()
results.append({
    "Model": "One-Class SVM",
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "Time (s)": end - start
})

# Isolation Forest 
print("\n Isolation Forest")
start = time.time()
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_train)
y_pred = iso.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)
end = time.time()
results.append({
    "Model": "Isolation Forest",
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "Time (s)": end - start
})

# LOF
print("\n Local Outlier Factor")
start = time.time()
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_train)
y_pred = lof.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1)
end = time.time()
results.append({
    "Model": "Local Outlier Factor",
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "Time (s)": end - start
})

results_df = pd.DataFrame(results)
print("\n Table")
print(results_df)

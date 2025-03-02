import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    L = int(lines[0].strip())
    N = int(lines[1].strip())
    data = []

    for line in lines[2:]:
        values = list(map(float, line.strip().split()))
        data.append(values)

    return np.array(data), N


def preprocess_data(data, N):
    X = data[:, :N]  # Pixel intensities
    attributes = data[:, N:]  # Attributes
    y = attributes[:, 2]  # Class/person number
    return X, y, attributes


def split_data_by_person(X, y, attributes):
    unique_classes = np.unique(y)
    X_train, X_test = [], []
    y_train, y_test = [], []

    for cls in unique_classes:
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)
        split_point = len(indices) // 2

        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        X_train.extend(X[train_indices])
        y_train.extend(y[train_indices])

        X_test.extend(X[test_indices])
        y_test.extend(y[test_indices])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Load datasets


x_data, N = load_data('x24x24.txt')
y_data, _ = load_data('y24x24.txt')
z_data, _ = load_data('z24x24.txt')

# Combine datasets
data = np.vstack((x_data, y_data, z_data))
X, y, attributes = preprocess_data(data, N)
X_train, X_test, y_train, y_test = split_data_by_person(X, y, attributes)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_test)

print("Random Forest Classifier Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nRandom Forest Classifier Report:\n", classification_report(y_test, rf_predictions))
joblib.dump(rf_classifier, 'rf_classifier.pkl')
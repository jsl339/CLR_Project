import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Data Loading and Preparation
import torch
import torchvision

# Load Fashion MNIST data
train_set = torchvision.datasets.FashionMNIST("./data", download=True)
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False)

# Convert data to numpy arrays
X_train = train_set.data.numpy()
labels_train = train_set.targets.numpy()
X_test = test_set.data.numpy()
labels_test = test_set.targets.numpy()

# Reshape the data and normalize
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2])) / 255.0
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2])) / 255.0

# Train-Test Split
X_train, X_val, labels_train, labels_val = train_test_split(X_train, labels_train, test_size=0.2, random_state=42)

# Initialize SVM model
svm = SVC(kernel='linear')

# Training
svm.fit(X_train, labels_train)

# Evaluation on Test Set
test_score = svm.score(X_test, labels_test)
print("Test Set Accuracy:", test_score)

# Classification Report on Validation Set
val_predictions = svm.predict(X_val)
print("Classification Report on Validation Set:")
print(classification_report(labels_val, val_predictions))

# Plotting confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions on test set
test_predictions = svm.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(labels_test, test_predictions)

# Calculate percentages
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix as percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".0%", cmap="Blues", annot_kws={"size": 12})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - SVM (Percentages)')
plt.show()

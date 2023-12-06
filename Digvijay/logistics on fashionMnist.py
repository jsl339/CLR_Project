import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Load Fashion MNIST dataset
fashion_mnist = fetch_openml(name='Fashion-MNIST')

# Split data into features and labels
X = fashion_mnist.data
y = fashion_mnist.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)

# Define hyperparameters for grid search
param_grid = [
    {'penalty': ['l1', 'l2'],
     'C': [0.01, 0.1, 1.0, 10, 100]}
]

# Grid Search with Cross Validation
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Model Evaluation on Test Set
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Calculate error rate
test_error = 1 - test_accuracy

# Print best hyperparameters and accuracy
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", grid_search.best_score_)

# Plotting confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions on test set
y_pred = best_model.predict(X_test_scaled)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate percentages
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix as percentages with zero decimal places
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".0%", cmap="Blues", annot_kws={"size": 12})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Logistic Regression (Percentages)')
plt.show()

# Plotting 
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Convert labels to binary format
y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Fit a OneVsRestClassifier on the best logistic regression model
ovr_classifier = OneVsRestClassifier(best_model)
ovr_classifier.fit(X_train_scaled, y_train_bin)

# Predict probabilities on the test set
y_score = ovr_classifier.predict_proba(X_test_scaled)

# Compute precision-recall pairs for each class
precision = dict()
recall = dict()
for i in range(len(np.unique(y_train))):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])

# Plot precision-recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(np.unique(y_train))):
    plt.plot(recall[i], precision[i], lw=2, label='Class {}'.format(i))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression (One-vs-Rest)')
plt.legend()
plt.grid()
plt.show()

# Plotting ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Binarize the labels
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# Fit the model
best_model.fit(X_train_scaled, y_train)

# Compute ROC curve and ROC area for each class
n_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], best_model.predict_proba(X_test_scaled)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])  # Adjust as needed for the number of classes
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fashion MNIST - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
dataset = pd.read_csv("Python for Rapid Engineering solutions/Project 1/heart1.csv")

# Separate the features and target variable
X = dataset.iloc[:, :-1]  # Taking 13 features
y = dataset.iloc[:, -1]

# Train-Test Split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the features
sc = StandardScaler()
sc.fit(train_X)
train_X_std = sc.transform(train_X)
test_X_std = sc.transform(test_X)

# Stack the whole dataset for standardized and non-standardized values
X_stack_std = np.vstack((train_X_std, test_X_std))
X_stack = np.vstack((train_X, test_X))
y_stack = np.hstack((train_y, test_y))

# Create a PERCEPTRON model
ppn = Perceptron(penalty='l1', max_iter=30, eta0=0.01, tol=0.1, fit_intercept=True)

# Fit the model
ppn.fit(train_X_std, train_y)

# Predictions and evaluation on the test set
ppn_y_pred = ppn.predict(test_X_std)
ppn_misclassified = (test_y != ppn_y_pred).sum()
ppn_accuracy = accuracy_score(test_y, ppn_y_pred)

# Prediction on the whole dataset
ppn_y_pred_stack = ppn.predict(X_stack_std)
ppn_misclassified_stack = (y_stack != ppn_y_pred_stack).sum()
ppn_accuracy_stack = accuracy_score(y_stack, ppn_y_pred_stack)


# LOGISTIC REGRESSION
lr = LogisticRegression(penalty='l2', tol=0.1, C=1, solver='newton-cholesky', max_iter=10, verbose=False, multi_class='ovr')
lr.fit(train_X_std, train_y)

# Predictions and evaluation on the test set
lr_y_pred_stack = lr.predict(test_X_std)
lr_misclassified = (test_y != lr_y_pred_stack).sum()
lr_accuracy = accuracy_score(test_y, lr_y_pred_stack)

# Prediction on the whole dataset
lr_y_pred_stack = lr.predict(X_stack_std)
lr_misclassified_stack = (y_stack != lr_y_pred_stack).sum()
lr_accuracy_stack = accuracy_score(y_stack, lr_y_pred_stack)


# SUPPORT VECTOR MACHINE
svm = SVC(kernel='rbf', gamma=0.1, tol=0.1, C=100, random_state=0)
svm.fit(train_X_std, train_y)

# Predictions and evaluation on the test set
svm_y_pred = svm.predict(test_X_std)
svm_misclassified = (test_y != svm_y_pred).sum()
svm_accuracy = accuracy_score(test_y, svm_y_pred)

# Predict on the whole dataset
svm_y_pred_stack = svm.predict(X_stack_std)
svm_misclassified_stack = (y_stack != svm_y_pred_stack).sum()
svm_accuracy_stack = accuracy_score(y_stack, svm_y_pred_stack)

# DECISION TREE CLASSIFIER
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
dtc.fit(train_X.values, train_y.values)

# Predictions and evaluation on the test set
dtc_y_pred = dtc.predict(test_X.values)
dtc_misclassified = (test_y != dtc_y_pred).sum()
dtc_accuracy = accuracy_score(test_y, dtc_y_pred)

# Prediction on the whole dataset
dtc_y_pred_stack = dtc.predict(X_stack)
dtc_misclassified_stack = (y_stack != dtc_y_pred_stack).sum()
dtc_accuracy_stack = accuracy_score(y_stack, dtc_y_pred_stack)

# RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier(n_estimators=24, criterion='entropy', random_state=0)
rfc.fit(train_X.values, train_y.values)

# Predictions and evaluation on the test set
rfc_y_pred = rfc.predict(test_X.values)
rfc_misclassified = (test_y != rfc_y_pred).sum()
rfc_accuracy = accuracy_score(test_y, rfc_y_pred)

# Prediction on the whole dataset
rfc_y_pred_stack = rfc.predict(X_stack)
rfc_misclassified_stack = (y_stack != rfc_y_pred_stack).sum()
rfc_accuracy_stack = accuracy_score(y_stack, rfc_y_pred_stack)

# K-NEAREST NEIGHBOR
knc = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
knc.fit(train_X_std, train_y)

# Predictions and evaluation on the test set
knc_y_pred = knc.predict(test_X_std)
knc_misclassified = (test_y != knc_y_pred).sum()
knc_accuracy = accuracy_score(test_y, knc_y_pred)

# Prediction on the whole dataset
knc_y_pred_stack = knc.predict(X_stack_std)
knc_misclassified_stack = (y_stack != knc_y_pred_stack).sum()
knc_accuracy_stack = accuracy_score(y_stack, knc_y_pred_stack)

# Table to compare the results of each ML Model
comparison = {
    'Model Name': ['Perceptron', 'Logistic Regression', 'Support Vector Classifier', 'Decision Tree Classifier', 'Random Forest Classifier', 'K-Neighbors Classifier'],
    'Accuracy': [ppn_accuracy_stack, lr_accuracy_stack, svm_accuracy_stack, dtc_accuracy_stack, rfc_accuracy_stack, knc_accuracy_stack]
}

comparison_table = pd.DataFrame(comparison)
print(comparison_table)

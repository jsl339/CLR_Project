import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

# Mute warnings
warnings.filterwarnings("ignore")

# Load Adult dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Read data into a DataFrame
data = pd.read_csv(url, header=None, na_values=' ?')

# Drop rows with NaN values
data.dropna(inplace=True)

# Preprocessing - separate features and target
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = (data.iloc[:, -1] == ' >50K').astype(int)  # Target (last column)

# Encoding categorical variables
X_encoded = pd.get_dummies(X)

# Convert column names to strings
X_encoded.columns = X_encoded.columns.astype(str)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with Kernels
svm = SVC()

# Define hyperparameters for grid search
param_grid = [
    {'kernel': ['linear'],
     'C': [10]
     }
]

# Grid Search with Cross Validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Model Evaluation on Test Set
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Print best hyperparameters and accuracy
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", grid_search.best_score_)

# Plotting confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Fit SVM on reduced data
best_model.fit(X_train_pca, y_train)

# Create a meshgrid for decision boundary plotting
h = .02  # step size in the mesh
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot data points
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title('Decision Boundary Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


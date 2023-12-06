import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

# Mute warnings
warnings.filterwarnings("ignore")

# Load Breast Cancer Wisconsin (Diagnostic) dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Read data into a DataFrame
data = pd.read_csv(url, header=None)

# Preprocessing - separate features and target
X = data.iloc[:, 2:]  # Features
y = data.iloc[:, 1]   # Target

# Encode categorical labels (M/B) into numerical labels (1/0)
y = (y == 'M').astype(int)

# Drop rows with NaN values
data.dropna(inplace=True)
X = data.iloc[:, 2:]
y = data.iloc[:, 1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with Kernels
svm = SVC()

# Define hyperparameters for grid search
param_grid = [
    {'kernel': ['linear', 'rbf', 'poly'],
     'C': [0.01, 0.1, 1.0, 10, 100]}
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

# Predictions on test set
y_pred = best_model.predict(X_test_scaled)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to percentages
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix as percentages using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, cmap="Blues", fmt=".2%", annot_kws={"size": 12})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix as Percentage - SVM')
plt.show()

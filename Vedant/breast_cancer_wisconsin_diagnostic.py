
# Install TensorFlow version 2.12.0
pip install tensorflow==2.12.0

# Import necessary libraries
import sys
!{sys.executable} -m pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the features
sc = StandardScaler()
sc_X_train = sc.fit_transform(X_train)
sc_X_test = sc.transform(X_test)

# Define a Keras model with hyperparameters
def create_model(learning_rate=0.01, activation='sigmoid', activation2='sigmoid',
                 units1=205, units2=194, output='sigmoid', dropout=0.4):

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=units1, activation=activation,
                                 input_dim=X_train.shape[1]))
    model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(units=units2, activation=activation2))
    model.add(keras.layers.Dense(units=1, activation=output))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Wrap in KerasClassifier with early stopping
model = KerasClassifier(build_fn=create_model, epochs=20)

# Define hyperparameter distributions to sample from
learning_rate = [0.001, 0.01, 0.1]
activation = ['relu', 'tanh', 'sigmoid']
activation2 = ['relu', 'sigmoid']
units1 = range(128, 500, 2)
units2 = range(32, 256, 2)
output = ['relu', 'sigmoid']
dropout = [0.1, 0.2, 0.3, 0.4]

distributions = dict(learning_rate=learning_rate, activation=activation,
                     activation2=activation2, units1=units1, units2=units2, output=output, dropout=dropout)

# Create randomized search with number of evaluations
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=distributions,
                                   n_iter=100, cv=3, n_jobs=-1)

# Fit the model on training data
random_search.fit(sc_X_train, y_train)

# Print best hyperparameters and score
print(random_search.best_params_)
print(random_search.best_score_)

# Import libraries for evaluation
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Extract best hyperparameters
best_params = random_search.best_params_

# Create a new model using the best hyperparameters
best_model = create_model(**best_params)

# Fit the best model on the training data
history = best_model.fit(sc_X_train, y_train, epochs=50, validation_split=0.2)

# Extract training history
train_history = history.history

# Plot accuracy vs. epoch
plt.plot(train_history['accuracy'], label='Training Accuracy')
plt.plot(train_history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on the test set
y_prob = best_model.predict(sc_X_test)
y_pred = (y_prob > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Create a heatmap for the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

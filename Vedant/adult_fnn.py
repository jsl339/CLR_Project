
# Install required libraries
pip install tensorflow==2.12.0

# Import necessary modules
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

# Fetch dataset from UCIML repository
data = fetch_ucirepo(id=2)

# Separate features and targets
X = data.data.features
y = data.data.targets

# Drop rows with missing values and update target values
cleaned_indices = X.dropna().index
X.dropna(inplace=True)
y = y.loc[cleaned_indices]
y.replace(("<=50K", "<=50K."), 0, inplace=True)
y.replace((">50K.", ">50K"), 1, inplace=True)

# Drop unnecessary columns
X.drop(["fnlwgt","education-num"], axis=1, inplace=True)

# Define categorical and numerical column lists
categorical_columns = ['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex']
numerical_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(transformers=[
        ('ord_encoder', OrdinalEncoder(), categorical_columns),
        ('std_scaler', StandardScaler(), numerical_columns)
    ])

# Create a preprocessing pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # Additional steps can be added to the pipeline
])

# Fit and transform the data
X_preprocessed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=0)

# Define early stopping for model training
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Function to create the FNN model for KerasClassifier
def create_model(optimizer='adam', units1=128, units2=64, units3=32, activation='relu', input_dim=X_train.shape[1]):
    model = Sequential()
    model.add(Dense(units=units1, activation=activation, input_dim=input_dim))
    model.add(Dense(units=units2, activation=activation))
    model.add(Dense(units=units3, activation=activation))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier wrapper
keras_classifier = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

# Parameter grid for RandomizedSearchCV
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'units1': range(64, 256, 2),
    'units2': range(32, 128, 2),
    'units3': range(16, 64, 2),
    'activation': ['relu', 'tanh']
}

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=keras_classifier, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the RandomizedSearchCV on the preprocessed data
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Import necessary modules for evaluation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Function to plot accuracy vs epoch
def plot_accuracy_vs_epoch(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to evaluate the model and get predictions
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_binary)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))

    # Calculate and print accuracy
    print(f'Test Accuracy: {test_accuracy}')

# Create and compile the model
model = create_model(**best_params)

# Fit the model on the training data with early stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping])

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Plot accuracy vs epoch
plot_accuracy_vs_epoch(history)

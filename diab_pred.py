import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")
df.head(5)

# Split the data into train (60%), validation (20%), and test (20%) sets
# .sample(frac=1) shuffles all the data randomly
# np.split divides the data at the specified indices
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    """
    Process a dataframe by:
    1. Separating features (X) and target (y)
    2. Scaling the features to similar ranges
    3. Optionally oversampling to balance classes
    4. Returning both combined and separated data
    """
    # Extract features (all columns except the last) and target (last column)
    X = dataframe[dataframe.columns[:-1]]
    y = dataframe['Outcome']
    
    # Scale features to make them equally important regardless of original range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # If requested, oversample the minority class to balance the dataset
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    # Combine features and target into a single array (for convenience)
    # reshape y from 1D to 2D column vector, then stack horizontally with X
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

# Process each dataset (only oversample the training data)
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# Create and train the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Set random seed for reproducible results
np.random.seed(42)

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")
df.head(5)

# Split the data into train (60%), validation (20%), and test (20%) sets
# .sample(frac=1) shuffles all the data randomly
# np.split divides the data at the specified indices
# Fixed random_state=42 ensures reproducible splits
train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))])

# It divides the data into 3 pieces:
# From index 0 to 60% (training data)
# From index 60% to 80% (validation data) 
# From index 80% to end (test data)

def scale_dataset(dataframe, oversample=False):
    """
    Process a dataframe by:
    1. Separating features (X) and target (y)
    2. Scaling the features to similar ranges using StandardScaler
    3. Optionally oversampling to balance classes using RandomOverSampler
    4. Returning both combined and separated data
    """
    # Extract features (all columns except the last) and target (last column)
    X = dataframe[dataframe.columns[:-1]]
    y = dataframe['Outcome']
    
    # Scale features to make them equally important regardless of original range
    # This is crucial for KNN since it uses distance calculations
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # If requested, oversample the minority class to balance the dataset
    if oversample:
        ros = RandomOverSampler(random_state=42)  # Fixed random state for reproducibility
        X, y = ros.fit_resample(X, y)  # Creates synthetic samples of minority class
    
    # Combine features and target into a single array (for convenience)
    # reshape y from 1D to 2D column vector, then stack horizontally with X
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

# Process each dataset (only oversample the training data to prevent data leakage)
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# Hyperparameter tuning: Find the optimal k value using validation set
print("Finding optimal k value...")
best_k = 0
best_recall = 0

# Test different k values systematically
for i in range(1, 100):  # Range for the k-values to test
    # Create and train KNN model with current k value
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    
    # Make predictions on validation set (not test set!)
    y_pred = knn_model.predict(X_valid)
    
    # Get detailed classification metrics
    report_dict = classification_report(y_valid, y_pred, output_dict=True)
    
    # Extract recall for class 1 (diabetes detection)
    # We optimize for recall because missing diabetes cases is more serious than false alarms
    recall_class1 = report_dict['1']['recall']
    
    # Track the k value that gives the best recall
    current_recall = recall_class1
    if(current_recall > best_recall):
        best_recall = current_recall
        best_k = i

print(f"Optimal k value: {best_k}")
print(f"Best recall achieved during validation: {best_recall:.3f}")

# Train final model using the optimal k value
print("\nTraining final model with optimal hyperparameters...")
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X_train, y_train)

# Make predictions on test set (final evaluation)
y_pred_final = final_knn_model.predict(X_test)

# Evaluate final model performance on unseen test data
print("\nFinal Model Performance on Test Set:")
print("="*50)
print(classification_report(y_test, y_pred_final))

# Additional performance summary
report_dict_final = classification_report(y_test, y_pred_final, output_dict=True)
print(f"\nModel Summary:")
print(f"- Algorithm: K-Nearest Neighbors")
print(f"- Optimal k: {best_k}")
print(f"- Test Accuracy: {report_dict_final['accuracy']:.1%}")
print(f"- Diabetes Detection Recall: {report_dict_final['1']['recall']:.1%}")
print(f"- Diabetes Detection Precision: {report_dict_final['1']['precision']:.1%}")
print(f"- Random Seed: 42 (for reproducibility)")
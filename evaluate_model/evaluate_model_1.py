import numpy as np
from PA_CS_1 import PA_CS
from load_data import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Define a function to evaluate the model
def evaluate_model(y_true, y_pred, eta_p, eta_n):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    
    weighted_sum = eta_p * sensitivity + eta_n * specificity
    return sensitivity, specificity, accuracy, weighted_sum

# Define a function for the grid search
def grid_search(X, y, initial_w, eta_values, n_splits=5):
    best_eta_p = None
    best_eta_n = None
    best_score = -np.inf
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for eta_p in eta_values:
        eta_n = 1 - eta_p
        
        sensitivities = []
        specificities = []
        accuracies = []
        weighted_sums = []
        
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Initialize the model weights
            model = {'w': initial_w.copy(), 'bias': False, 'p_kernel_degree': 1}
            
            # Train the model using the training data
            for t in range(X_train.shape[0]):
                model, _, _ = PA_CS(y_train[t], X_train[t:t+1], model, eta_p, eta_n)
            
            # Make predictions on the validation data
            f_val = np.dot(X_val, model['w'].T)
            y_pred = np.where(f_val >= 0, 1, -1).flatten()
            
            # Evaluate the model
            sensitivity, specificity, accuracy, weighted_sum = evaluate_model(y_val, y_pred, eta_p, eta_n)
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)
            weighted_sums.append(weighted_sum)
        
        # Average performance metrics over all folds
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
        avg_accuracy = np.mean(accuracies)
        avg_weighted_sum = np.mean(weighted_sums)
        
        # Use the weighted sum for model selection
        score = avg_weighted_sum
        
        if score > best_score:
            best_score = score
            best_eta_p = eta_p
            best_eta_n = eta_n
    
    return best_eta_p, best_eta_n, best_score

# Example dataset and parameters
# Replace with your actual data
X, y, n = load_data('data/binary_synthetic_data.libsvm', 'libsvm', 'bc')
print(X.shape)

initial_w = np.zeros((1, X.shape[1]))  # Initial weight vector

eta_values = np.linspace(0.05, 0.9, 20)

# Perform the grid search
best_eta_p, best_eta_n, best_score = grid_search(X, y, initial_w, eta_values)

print(f"Best eta_p: {best_eta_p}, Best eta_n: {best_eta_n}, Best score: {best_score}")

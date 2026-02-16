import numpy as np
from PA_CS_2 import PA_CS
from load_data import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

# Define a function to evaluate the model based on weighted misclassification cost
def evaluate_model(y_true, y_pred, cp, cn):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]

    # Calculate misclassification costs
    cost = cp * FN + cn * FP

    return cost

# Define a function for the grid search to find optimal cp and cn
def grid_search(X, y, initial_w, eta_values, cp_values, cn_values, n_splits=5):
    best_cp = None
    best_cn = None
    best_score = np.inf

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for cp in cp_values:
        for cn in cn_values:
            sensitivities = []
            specificities = []
            costs = []

            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Initialize the model weights
                model = {'w': initial_w.copy(), 'bias': False, 'p_kernel_degree': 1}

                # Train the model using the training data
                for t in range(X_train.shape[0]):
                    model, _, _ = PA_CS(y_train[t], X_train[t:t+1], model, cp, cn)

                # Make predictions on the validation data
                f_val = np.dot(X_val, model['w'].T)
                y_pred = np.where(f_val >= 0, 1, -1).flatten()

                # Evaluate the model
                cost = evaluate_model(y_val, y_pred, cp, cn)

                costs.append(cost)

            # Average misclassification cost over all folds
            avg_cost = np.mean(costs)

            if avg_cost < best_score:
                best_score = avg_cost
                best_cp = cp
                best_cn = cn

    return best_cp, best_cn, best_score

# Example dataset and parameters
# Replace with your actual data
X, y, n = load_data('data/binary_synthetic_data.libsvm', 'libsvm', 'bc')
print(X.shape)

initial_w = np.zeros((1, X.shape[1]))  # Initial weight vector

cp_values = np.linspace(0.05, 0.95, 19)
cn_values = 1 - cp_values

# Perform the grid search
best_cp, best_cn, best_score = grid_search(X, y, initial_w, cp_values, cp_values, cn_values)

print(f"Best cp: {best_cp}, Best cn: {best_cn}, Best score: {best_score}")

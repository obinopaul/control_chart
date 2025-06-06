# pa_algorithm.py
import numpy as np

def CPA2(y_t, x_t, model, eta_p=None, eta_n=None, ratio_Tp_Tn=None, cost_matrix=None, variant="PA-II"):
    """
    Cost-Sensitive Passive-Aggressive (PA) Algorithm.
    
    Parameters:
        y_t:         Class label of the t-th instance (1 or -1).
        x_t:         t-th training data instance (X(t,:)).
        model:       A struct of the weight vector (w).
        eta_p:       Positive class weight (not used in this implementation).
        eta_n:       Negative class weight (not used in this implementation).
        ratio_Tp_Tn: Ratio of positive to negative samples (not used in this implementation).
        cost_matrix: A dictionary defining the cost ρ(y_t, hat_y_t).
        variant:     The PA variant to use ('PA', 'PA-I', 'PA-II').
        C:           Regularization parameter for PA-I and PA-II variants.
    
    Returns:
        model:      Updated model with weight vector.
        hat_y_t:    Predicted class label for the current instance.
        l_t:        Suffered cost-sensitive loss.
    """
    # Initialize weight vector
    w = model.w
    C = model.C

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1  # Predicted label

    # Use cost matrix if provided, otherwise default cost is 1 for misclassification
    if cost_matrix:
        cost = cost_matrix[y_t][hat_y_t]  # Access the cost from the cost matrix
    else:
        cost = 1 if y_t != hat_y_t else 0   # Default cost

    # Check if prediction is correct
    if y_t == hat_y_t:
        # No loss for correct predictions
        l_t = 0
    else:
        # Only calculate cost-sensitive loss for incorrect predictions (FP, FN)
        l_t = np.abs(np.dot(w, x_t.T)) + np.sqrt(cost)  # Absolute dot product + sqrt(cost)
    
    # Update only if there is a loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2 + 1e-10  # Adding small epsilon to avoid division by zero

        # Step size calculation depending on the PA variant
        if variant == "PA":
            gamma_t = l_t / s_t  # Standard PA variant
        
        elif variant == "PA-I":
            gamma_t = min(C, l_t / s_t)  # PA-I variant with a cap on the step size
        
        elif variant == "PA-II":
            gamma_t = l_t / (s_t + 1/(2*C))  # PA-II variant with regularization term
        
        # Update the weight vector
        model.w = w + gamma_t * y_t * x_t # Update the weight vector

    return model, hat_y_t, l_t
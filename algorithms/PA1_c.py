# pa_algorithm.py
import numpy as np

def PA1_c(y_t, x_t, model, eta_p=None, eta_n=None, ratio_Tp_Tn=None, cost_matrix=None, variant="PA-I"):
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
        cost = 1  # Default cost

    # Cost-sensitive margin
    margin = np.sqrt(cost)
    l_t = max(0, margin - y_t * f_t)  # Cost-sensitive hinge loss
    
    # Update only if there is a loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2

        # Step size calculation depending on the PA variant
        if variant == "PA":
            gamma_t = l_t / s_t if s_t > 0 else 1  # Standard PA variant
        
        elif variant == "PA-I":
            gamma_t = min(C, l_t / s_t) if s_t > 0 else C  # PA-I variant with cap
        
        elif variant == "PA-II":
            gamma_t = l_t / (s_t + 1/(2*C)) if s_t > 0 else 1/(2*C)  # PA-II variant with regularization
        
        # Update the weight vector
        model.w = w + gamma_t * y_t * x_t

    return model, hat_y_t, l_t

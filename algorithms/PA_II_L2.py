import numpy as np

def PA_II_L2(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn, cost_matrix=None):
    """
    PA2: Cost-Sensitive Passive-Aggressive (PA-II) learning algorithm
    
    Parameters:
    y_t : int
        Class label of the t-th instance.
    x_t : numpy array
        The t-th training data instance.
    model : object
        Classifier model containing weight vector, regularization parameter C, bias, and polynomial kernel degree.
    eta_p : float
        Cost-sensitive parameter for the positive class.
    eta_n : float
        Cost-sensitive parameter for the negative class.
    T_p : int
        Number of positive examples.
    T_n : int
        Number of negative examples.

    Returns:
    model : object
        Updated model containing the weight vector.
    hat_y_t : int
        Predicted class label.
    l_t : float
        Suffered loss.
    """

    # Initialization
    w = model.w
    C = model.C
    
    # Compute rho for maximizing weighted sum of sensitivity and specificity
    rho = (eta_p / eta_n) * (1 / ratio_Tp_Tn)

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1

    # Cost-Sensitive Hinge Loss Type II
    l_t = (rho if y_t == 1 else 1) * max(0, 1 - y_t * f_t)

    # Update on non-zero loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2
               
        # Correct computation of tau_t (gamma_t in code)
        rho_term = rho if y_t == 1 else 1
        numerator = l_t 
        denominator = (rho_term ** 2) * s_t + (1 / (2 * C))
        
        gamma_t = numerator / denominator

        # w = w + gamma_t * y_t * x_t
        # Update the weight vector
        w = w + gamma_t * rho_term * y_t * x_t
    
    # Save the updated weight back to the model
    model.w = w

    return model, hat_y_t, l_t

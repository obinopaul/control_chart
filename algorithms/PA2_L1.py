import numpy as np

def PA2_L1(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative):
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


    
    # Calculate class-specific regularization parameters
    C_pos = C / num_positive  # Regularization for the positive class
    C_neg = C / num_negative  # Regularization for the negative class

    # Compute rho for maximizing weighted sum of sensitivity and specificity
    rho = (eta_p * num_negative) / (eta_n * num_positive)  # Cost-sensitive parameter

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1

    # Cost-Sensitive Hinge Loss Type I
    l_t = float(max(0, (rho if y_t == 1 else 1) - y_t * f_t)) # Hinge Loss I

    # Update on non-zero loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2

        # Use C_pos if y_t is positive, otherwise use C_neg
        C = C_pos if y_t == 1 else C_neg
        
        gamma_t = l_t / (s_t + (1 / (2 * C)))  # PA-II: includes quadratic penalty

        # Update the weight vector
        w = w + gamma_t * y_t * x_t
    
    # Save the updated weight back to the model
    model.w = w

    return model, hat_y_t, l_t

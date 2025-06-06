import numpy as np

import numpy as np

def PA_L2(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn):
    """
    PA: Cost-Sensitive Passive-Aggressive (PA) learning algorithms
    
    Parameters:
    y_t : int
        Class label of the t-th instance.
    x_t : numpy array
        The t-th training data instance.
    model : object
        Classifier model containing weight vector, bias, and polynomial kernel degree.
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

    # Compute rho for maximizing weighted sum of sensitivity and specificity
    rho = (eta_p / eta_n) * (1 / ratio_Tp_Tn) # Cost-sensitive parameter 
    
    # Prediction
    f_t = np.dot(w, x_t)
    hat_y_t = 1 if f_t >= 0 else -1

    # Cost-Sensitive Hinge Loss Type II
    l_t = (rho if y_t == 1 else 1) * max(0, 1 - y_t * f_t)

    # Update on non-zero loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2
        if s_t > 0:
            gamma_t = l_t / s_t  # Step size (PA variant)
        else:
            gamma_t = 1  # Special case when the norm of x_t is zero.

        # Update the weight vector
        model.w = w + gamma_t * y_t * x_t

    return model, hat_y_t, l_t 

import numpy as np

def CSOGD_1(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn, method='sum'):
    """
    Cost-Sensitive Online Gradient Descent (CSOGD) Algorithm.

    Parameters:
    y_t: scalar
        Target label (-1 or +1) for the current instance.
    x_t: numpy array of shape (n_features,)
        Feature vector for the current instance.
    model: object
        Model containing the learning rate and other parameters.
    eta_p: float
        Weight assigned to sensitivity.
    eta_n: float
        Weight assigned to specificity.
    T_p: int
        Number of positive examples.
    T_n: int
        Number of negative examples.
    method: str, optional
        Method to use, either "sum" for maximizing weighted sum of sensitivity and specificity
        or "cost" for minimizing weighted misclassification cost. Default is "sum".

    Returns:
    w: numpy array of shape (n_features,)
        Updated weight vector after training.
    """

    # Initialize weight vector
    w = model.w

    # Calculate rho
    if method == 'sum':
        rho = (eta_p / eta_n) * (1 / ratio_Tp_Tn)
    elif method == 'cost':
        rho = eta_p / eta_n
    else:
        raise ValueError("Invalid method. Choose either 'sum' or 'cost'.")

    # Predict the label
    y_hat_t = np.sign(np.dot(w, x_t))
   
    # Cost-Sensitive Hinge Loss Type I
    l_t = float(max(0, (rho if y_t == 1 else 1) - y_t * np.dot(w, x_t))) # Hinge Loss I

    # Update the classifier if loss is greater than 0
    if l_t > 0:
        gradient = -y_t * x_t
        w = w - model.C / np.sqrt(model.t + 1) * gradient  # Decaying learning rate

    # Update the model's weight
    model.w = w
    model.t += 1  # Increment the iteration counter

    return model, y_hat_t, l_t


import numpy as np
from math import log, exp

def OGD_1(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn, cost_matrix=None):
    # OGD: Online Gradient Descent (OGD) algorithms with cost-sensitive hinge loss
    #--------------------------------------------------------------------------
    # Reference:
    # - Martin Zinkevich. Online convex programming and generalized infinitesimal 
    # gradient ascent. In ICML, pages 928–936, 2003.
    #--------------------------------------------------------------------------
    # INPUT:
    #      y_t:     class label of t-th instance;
    #      x_t:     t-th training data instance, e.g., X(t,:);
    #    model:     classifier
    #    eta_p:    Cost-sensitive parameter for the positive class
    #    eta_n:    Cost-sensitive parameter for the negative class
    #    T_p:      Number of positive examples
    #    T_n:      Number of negative examples
    #
    # OUTPUT:
    #    model:     a struct of the weight vector (w) and the SV indexes
    #  hat_y_t:     predicted class label
    #      l_t:     suffered loss

    # Initialization
    w = model.w
    loss_type = model.loss_type  # type of loss
    eta = model.C  # learning rate

    # Calculate cost-sensitive parameter rho
    rho = (eta_p / eta_n) * (1 / ratio_Tp_Tn)

    # Prediction
    f_t = np.dot(w, x_t.T)  # Use dot product for 1D arrays
    f_t = float(f_t)  # Convert to scalar
    hat_y_t = 1 if f_t >= 0 else -1

    # Making Update
    eta_t = eta / np.sqrt(model.t)  # learning rate = eta*(1/sqrt(t)) this learning rate decays over time

    # 0 - 1 Loss
    if loss_type == 0:
        l_t = (hat_y_t != y_t)  # 0 - correct prediction, 1 - incorrect
        if l_t > 0:
            w += eta_t * y_t * x_t  # Update w with hinge loss derivative

    # Cost-sensitive Hinge Loss
    elif loss_type == 1:
        # l_t = max(0, (rho if y_t == 1 else 1) - y_t * f_t)  # Cost-sensitive Hinge Loss I
            # Cost-Sensitive Hinge Loss Type I
        l_t = float(max(0, (rho if y_t == 1 else 1) - y_t * f_t)) # Hinge Loss I
        if l_t > 0:
            w += eta_t * y_t * x_t  # Update w with hinge loss derivative

    # Logistic Loss
    elif loss_type == 2:
        l_t = log(1 + exp(-y_t * f_t))
        if l_t > 0:
            w += eta_t * y_t * x_t * (1 / (1 + exp(y_t * f_t)))  # Update w with log loss derivative

    # Square Loss
    elif loss_type == 3:
        l_t = 0.5 * (y_t - f_t) ** 2
        if l_t > 0:
            w += -eta_t * (f_t - y_t) * x_t  # Update w with square loss derivative

    else:
        print('Invalid loss type.')

    model.w = w
    model.t = model.t + 1  # iteration counter
    
    return model, hat_y_t, l_t

import numpy as np
from  math import log, exp
def OGD(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn, cost_matrix=None):
    # OGD: Online Gradient Descent (OGD) algorithms
    #--------------------------------------------------------------------------
    # Reference:
    # - Martin Zinkevich. Online convex programming and generalized infinitesimal 
    # gradient ascent. In ICML, pages 928?36, 2003.
    #--------------------------------------------------------------------------
    # INPUT:
    #      y_t:     class label of t-th instance;
    #      x_t:     t-th training data instance, e.g., X(t,:);
    #    model:     classifier
    #
    # OUTPUT:
    #    model:     a struct of the weight vector (w) and the SV indexes
    #  hat_y_t:     predicted class label
    #      l_t:     suffered loss

    # Initialization
    w           = model.w
    loss_type   = model.loss_type           # type of loss
    eta         = model.C                   # learning rate

    # Prediction
    f_t = np.dot(w, x_t.T)  # Use dot product for 1D arrays
    f_t = float(f_t)  # Convert to scalar
    hat_y_t = 1 if f_t >= 0 else -1 

    # Making Update
    eta_t   = eta/np.sqrt(model.t)                              # learning rate = eta*(1/sqrt(t)) this learning rate decays over time

    # 0 - 1 Loss
    if loss_type == 0:
        pass 

    # Hinge Loss
    elif loss_type == 1:
        l_t = max(0,1-y_t*f_t) 
        if l_t > 0:
            w += eta_t * y_t * x_t  # Update w with hinge loss derivative

    # Logistic Loss
    elif loss_type == 2:
        pass 
            
    # Square Loss
    elif loss_type == 3:
        pass

    else:
        print('Invalid loss type.')

    model.w = w
    model.t = model.t + 1  # iteration counter

    return model, hat_y_t, l_t 

import numpy as np
def PA1(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative):
    # PA1: Passive-Aggressive (PA) learning algorithms (PA-I variant)
    #--------------------------------------------------------------------------
    # Reference:
    # - Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram
    # Singer. Online passive-aggressive algorithms. JMLR, 7:551?85, 2006.
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
    C           = model.C

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1 

    # Hinge Loss
    l_t = max(0,1-y_t*f_t)
    
    # Update on non-zero loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2
        # Set a default value for gamma_t in case s_t is zero
        gamma_t = 0.0
        if s_t > 0:
            gamma_t = min(C, l_t / s_t)  # PA-I: bounded by C
        else:
            gamma_t = C  # Special case to avoid division by 0

        # Update the weight vector
        model.w = w + gamma_t * y_t * x_t 
    
    return model, hat_y_t, l_t

import numpy as np
def PA2_Csplit(y_t, x_t, model, eta_p, eta_n, ratio_Tp_Tn, cost_matrix=None, variant = "PA-II"):
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
    w = model.w # Weight vector
    C = model.C # Regularization parameter for PA-I and PA-II variants
    # C_pos = model.C_pos # Regularization parameter for positive class 
    # C_neg = model.C_neg # Regularization parameter for negative class

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1 

    n_pos=100
    n_neg=900
    
    # Calculate class-specific regularization parameters
    C_pos = C / n_pos  # Regularization for the positive class
    C_neg = C / n_neg  # Regularization for the negative class

    # Hinge Loss
    l_t = max(0,1-y_t*f_t)
    
    # Update only if there is a loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2 + 1e-10  # Adding small epsilon to avoid division by zero

        # Use C_pos if y_t is positive, otherwise use C_neg
        C = C_pos if y_t == 1 else C_neg
        
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

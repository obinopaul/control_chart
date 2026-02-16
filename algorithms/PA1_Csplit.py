import numpy as np
def PA1_Csplit(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative, variant = "PA-I"):

    # Initialization
    w = model.w # Weight vector
    C = model.C # Regularization parameter for PA-I and PA-II variants

    # Prediction
    f_t = np.dot(w, x_t.T)
    hat_y_t = 1 if f_t >= 0 else -1 


    
    # Calculate class-specific regularization parameters
    C_pos = C / num_positive  # Regularization for the positive class
    C_neg = C / num_negative  # Regularization for the negative class
    
    # Hinge Loss
    l_t = max(0,1-y_t*f_t)
    
    # Update only if there is a loss
    if l_t > 0:
        s_t = np.linalg.norm(x_t) ** 2 + 1e-10  # Adding small epsilon to avoid division by zero

        # Use C_pos if y_t is positive, otherwise use C_neg
        C = C_pos if y_t == 1 else C_neg

        gamma_t = min(C, l_t / s_t)  # PA-I variant with a cap on the step size
 
        # Update the weight vector
        model.w = w + gamma_t * y_t * x_t # Update the weight vector

    return model, hat_y_t, l_t

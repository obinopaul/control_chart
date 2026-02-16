import numpy as np

def CSTG_1(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative):
    """
    CSTG-I: Cost-sensitive Sparse Online Learning via Truncated Gradient (Type I)
    
    Reference:
    - Zhong Chen et al. "CSTG: An Effective Framework for Cost-sensitive 
      Sparse Online Learning", SDM 2017.
    """
    
    # -------------------------------------------------------------------------
    
    # Local variables
    w = model.w
    t = model.t
    g_param = model.g_param
    theta = model.theta_param
    K = model.K_param
    C = model.C_param
    eta_base = model.eta_base
    
    # Calculate learning rate eta_t = eta / sqrt(t)
    eta_t = eta_base / np.sqrt(t)
    
    # Prediction
    # x_t is (1, d), w is (1, d)
    f_t = np.dot(w, x_t.T)
    f_t = float(f_t)
    hat_y_t = 1 if f_t >= 0 else -1
    
    # Parameters for Cost (Static Kappa)
    # Algorithm 1 Input: kappa = mu_1 / mu_2
    kappa = eta_p / eta_n if eta_n != 0 else 1.0
    
    # Cost-Sensitive Hinge Loss Type I (Eq 3.1)
    # l^I = max(0, kappa_bar - y * w * x)
    # where kappa_bar = kappa * I(y=1) + I(y=-1)
    kappa_bar = kappa if y_t == 1 else 1.0
    
    l_t = float(max(0, kappa_bar - y_t * f_t))
    
    # Update Rule (Algorithm 1, Lines 6-17)
    if l_t > 0:
        # Gradient of Loss I: -y_t * x_t
        # Note: The Objective (Eq 3.2) is weighted by C.
        # Standard SGD on C*Loss implies gradient is C * (-y_t * x_t)
        gradient = -y_t * x_t
        
        # Standard Gradient Descent Step (Un-truncated)
        # w_tmp = w - eta_t * (C * gradient)
        # Algorithm 1 Line 10 simplifies this to "w + eta_t y_t x_t" (ignoring C visually),
        # but Theorem 4.1 bounds depend on C. We include C for correctness with Eq 3.2.
        w_update_term = -eta_t * C * gradient # = eta_t * C * y_t * x_t
        
        # Apply Truncation Logic (Algorithm 1 Lines 10-15)
        # Note: The paper Eq 3.3 implies truncation happens if K|t.
        # The pseudocode nests it inside the update. We apply logic per element.
        
        w_next = np.zeros_like(w)
        w_tmp = w + w_update_term
        
        # Flatten for iteration
        w_tmp_flat = w_tmp.flatten()
        w_next_flat = np.zeros_like(w_tmp_flat)
        x_flat = x_t.flatten() # Not used in truncation conditional check explicitly in code below
        
        # Vectorized implementation of Lines 10-15
        for i in range(len(w_tmp_flat)):
            val = w_tmp_flat[i]
            
            # Check if truncation applies this step
            if t % K == 0:
                # Line 10: if w in [0, theta]
                if 0 <= val <= theta:
                    # Line 11: max(0, val - eta_t * g)
                    w_next_flat[i] = max(0, val - eta_t * g_param)
                
                # Line 12: else if w in [-theta, 0]
                elif -theta <= val <= 0:
                    # Line 13: min(0, val + eta_t * g)
                    w_next_flat[i] = min(0, val + eta_t * g_param)
                
                # Line 14: else
                else:
                    w_next_flat[i] = val
            else:
                # If not K-th step, just standard update (Line 15 logic equivalent/fallback)
                w_next_flat[i] = val
                
        model.w = w_next_flat.reshape(1, -1)
        
    # else: w_{t+1} = w_t (Implicit)

    return model, hat_y_t, l_t
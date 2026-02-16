import numpy as np

def CSTG_2(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative):
    """
    CSTG-II: Cost-sensitive Sparse Online Learning via Truncated Gradient (Type II)
    
    Reference:
    - Zhong Chen et al. "CSTG: An Effective Framework for Cost-sensitive 
      Sparse Online Learning", SDM 2017.
    """
    # -------------------------------------------------------------------------
    
    w = model.w
    t = model.t
    g_param = model.g_param
    theta = model.theta_param
    K = model.K_param
    C = model.C_param
    eta_base = model.eta_base
    
    eta_t = eta_base / np.sqrt(t)
    
    f_t = np.dot(w, x_t.T)
    f_t = float(f_t)
    hat_y_t = 1 if f_t >= 0 else -1
    
    # Parameters for Cost
    kappa = eta_p / eta_n if eta_n != 0 else 1.0
    
    # Cost-Sensitive Hinge Loss Type II (Eq 3.1)
    # l^II = kappa_bar * max(0, 1 - y * w * x)
    # Note: For Type II, kappa_bar is a multiplier OUTSIDE the max
    kappa_bar = kappa if y_t == 1 else 1.0
    
    loss_raw = max(0, 1.0 - y_t * f_t)
    l_t = float(kappa_bar * loss_raw)
    
    # Update Rule (Algorithm 1, Lines 18-29)
    if l_t > 0:
        # Gradient of Loss II: -y_t * kappa_bar * x_t
        # (Line 20 in Algorithm 1)
        gradient = -y_t * kappa_bar * x_t
        
        # Apply C (from Eq 3.2)
        w_update_term = -eta_t * C * gradient
        
        w_tmp = w + w_update_term
        w_tmp_flat = w_tmp.flatten()
        w_next_flat = np.zeros_like(w_tmp_flat)
        
        for i in range(len(w_tmp_flat)):
            val = w_tmp_flat[i]
            
            if t % K == 0:
                # Line 22: if w in [0, theta]
                if 0 <= val <= theta:
                    w_next_flat[i] = max(0, val - eta_t * g_param)
                # Line 24: else if w in [-theta, 0]
                elif -theta <= val <= 0:
                    w_next_flat[i] = min(0, val + eta_t * g_param)
                else:
                    w_next_flat[i] = val
            else:
                w_next_flat[i] = val
                
        model.w = w_next_flat.reshape(1, -1)

    return model, hat_y_t, l_t
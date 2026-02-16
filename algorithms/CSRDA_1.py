import numpy as np


def CSRDA_1(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative):
    """
    CSRDA-I: Cost-sensitive Regularized Dual Averaging (Type I) learning algorithm
    
    Reference:
    - Zhong Chen et al. CSRDA: Cost-sensitive Regularized Dual Averaging for 
      Handling Imbalanced and High-dimensional Streaming Data. IEEE ICBK, 2021.
    
    INPUT:
         y_t:     class label of t-th instance (y_t ∈ {-1, +1})
         x_t:     t-th training data instance, e.g., X(t,:)
       model:     classifier with weight vector w, average gradient g_bar, 
                  and parameters lambda_param, gamma_param, L
       eta_p:     Cost-sensitive parameter for the positive class (paper uses 0.95)
       eta_n:     Cost-sensitive parameter for the negative class (paper uses 0.05)
    num_positive: Number of positive examples (not used - for API consistency)
    num_negative: Number of negative examples (not used - for API consistency)
    
    OUTPUT:
       model:     updated model with weight vector (w) and average gradient (g_bar)
     hat_y_t:     predicted class label
         l_t:     suffered loss
    
    Algorithm Details:
    - Loss Function (Eq. 1): ℓ_I(w; (x,y)) = max(0, ᾱ - yw^T x)
    - Gradient Update (Eq. 7): ḡ_t = (t-1)/t · ḡ_{t-1} - ᾱ_t·y_t/t · x_t (if loss > 0)
    - Weight Update (Eq. 6): w^(i)_{t+1} = 0 if |ḡ^(i)_t| ≤ λ, else -√t/γ · (ḡ^(i)_t - λ·sign(ḡ^(i)_t))
    - Dynamic κ_t (Algorithm 1, Line 5): κ_t = (count_neg + 1) / (count_pos + 1) over sliding window L
    """
    
    # Initialization
    w = model.w
    g_bar = model.g_bar
    lambda_param = model.lambda_param  # λ for sparsity (paper: 10^-1)
    gamma_param = model.gamma_param    # γ for smoothness (paper: 10^-3)
    L = model.L                         # Sliding window length (paper: 100)
    
    # eta_p = 0.95
    # eta_n = 0.05
    
    
    # Prediction
    f_t = np.dot(w, x_t.T)  # Use dot product for 1D arrays
    f_t = float(f_t)         # Convert to scalar
    hat_y_t = 1 if f_t >= 0 else -1
    
    # Update sliding window for dynamic imbalance ratio κ_t
    # Algorithm 1, Line 5: κ_t = (Σ I(y_r=-1) + 1) / (Σ I(y_r=+1) + 1) over last L instances
    model.window_labels.append(y_t)
    if len(model.window_labels) > L:
        model.window_labels.pop(0)
    
    # Count positive and negative labels in window (with +1 smoothing)
    count_negative = sum(1 for y in model.window_labels if y == -1) + 1
    count_positive = sum(1 for y in model.window_labels if y == +1) + 1
    

    # Paper Algorithm 1, Line 5: κ_t = κ * (ratio)
    # Where κ = mu_+ / mu_- (eta_p / eta_n)
    kappa_static = eta_p / eta_n
    kappa_t = kappa_static * (count_negative / count_positive)
    
    
    # kappa_t = count_negative / count_positive  # Dynamic imbalance ratio
    
    # Compute cost parameter ᾱ_t (Algorithm 1, Line 5)
    # ᾱ_t = κ_t·I(y_t=+1) + I(y_t=-1)
    # This ᾱ_t is used as the margin in Loss Type I
    alpha_t = kappa_t if y_t == 1 else 1.0
    
    # Cost-Sensitive Hinge Loss Type I (Equation 1)
    # ℓ_I(w; (x,y)) = max(0, ᾱ - yw^T x)
    l_t = float(max(0, alpha_t - y_t * f_t))
    
    # Gradient Update (Equations 7 and 6)
    if l_t > 0:
        # Algorithm 1, Line 10: update ḡ_t = (t-1)/t · ḡ_{t-1} - ᾱ_t·y_t/t · x_t
        # g_bar = ((model.t - 1) / model.t) * g_bar - (alpha_t * y_t / model.t) * x_t

        # Note: CSRDA-I gradient does NOT include alpha_t/kappa_t as a multiplier.
        g_bar = ((model.t - 1) / model.t) * g_bar - (y_t / model.t) * x_t
        
    else:
        # Algorithm 1, Line 13: update ḡ_t = (t-1)/t · ḡ_{t-1}
        g_bar = ((model.t - 1) / model.t) * g_bar
    
    # Weight Update according to Equation (6)
    # w^(i)_{t+1} = 0, if |ḡ^(i)_t| ≤ λ
    # w^(i)_{t+1} = -√t/γ · (ḡ^(i)_t - λ·sign(ḡ^(i)_t)), otherwise
    g_bar_flat = g_bar.flatten()
    w_new = np.zeros_like(g_bar_flat)
    
    for i in range(len(g_bar_flat)):
        if abs(g_bar_flat[i]) <= lambda_param:
            w_new[i] = 0  # Sparse solution
        else:
            w_new[i] = -(np.sqrt(model.t) / gamma_param) * (
                g_bar_flat[i] - lambda_param * np.sign(g_bar_flat[i])
            )
    
    # Update model
    model.w = w_new.reshape(1, -1)
    model.g_bar = g_bar
    model.t = model.t + 1  # iteration counter
    
    return model, hat_y_t, l_t
import numpy as np

def PA_CS(y_ts, x_ts, model, eta_p, eta_n):
    """
    PA: Cost-Sensitive Passive-Aggressive (PA) learning algorithms
    Reference:
    - Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram Singer.
      Online passive-aggressive algorithms. JMLR, 7:551–585, 2006.

    INPUT:
         y_ts:    class labels of t-th instances;
         x_ts:    t-th training data instances, e.g., X(t,:);
       model:     classifier
      eta_p:    Cost-sensitive parameter for positive class
      eta_n:    Cost-sensitive parameter for negative class

    OUTPUT:
      model:     a struct of the weight vector (w) and the SV indexes
    hat_y_ts:    predicted class labels
         l_ts:    suffered losses
    """
    # Initialization
    w = model['w']
    bias = model.get('bias', False)
    degree = model.get('p_kernel_degree', 1)  # Polynomial kernel degree

    # Transform input vectors
    if degree > 1:
        poly = model['poly']
        x_ts = poly.fit_transform(x_ts)  # Polynomial feature mapping for x_ts

    # Add bias term in feature vectors
    elif bias:
        x_ts = np.concatenate((np.ones((x_ts.shape[0], 1)), x_ts), axis=1)

    # Prediction
    f_ts = np.dot(x_ts, w.T)
    hat_y_ts = np.where(f_ts >= 0, 1, -1).flatten()

    # Hinge Loss
    rho = (eta_p if y_ts == 1 else eta_n)   # rho is the cost parameter
    l_ts = np.maximum(0, rho - y_ts * f_ts)  # Cost-sensitive hinge loss

    # Update on non-zero loss
    s_ts = np.linalg.norm(x_ts, axis=1)**2
    gamma_ts = l_ts / s_ts  # PA-I

    w_update = np.sum((gamma_ts * y_ts)[:, None] * x_ts, axis=0).reshape(w.shape)
    w += w_update

    model['w'] = w

    return model, hat_y_ts, l_ts


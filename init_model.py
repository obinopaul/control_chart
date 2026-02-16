import numpy as np

class Model:

    def __init__(self, options, d, nb_class):
        # INPUT:
        #  options:     method name and setting
        #        d:     data dimensionality
        # nb_class:     number of class labels

        UPmethod = options.method.upper()
        self.C = getattr(options, 'C', None)  # Default initialization of C
        
        if options.task_type == 'bc':
            self.task_type = 'bc'

            if UPmethod in ['GAUSSIAN_KERNEL_PERCEPTRON', 'GAUSSIAN_KERNEL_OGD']:
                # Initialize parameters for kernel methods
                self.max_sv = options.max_sv                # Number of instances to keep for kernel approach
                self.alpha = np.zeros(self.max_sv)          # Weights corresponding to each of the support vectors
                self.SV = np.zeros((self.max_sv, d))        # Support vector array with values of x
                self.sv_num = 0                             # Number of support vectors added so far
                self.kernel = options.kernel                # Kernel method to use
                self.sigma = options.sigma                  # Hyperparameter for Gaussian kernel
                self.index = 0                              # Index for budget maintenance
                self.C = getattr(options, 'C', None)        # Ensure C exists

                if UPmethod == 'GAUSSIAN_KERNEL_OGD':
                    self.t = 1                              # Iteration number
                    self.loss_type = options.loss_type      # Loss type
                    self.C = options.C                      # Regularization parameter

            else:
                # Initialize weight vector for non-kernel methods
                self.w = np.zeros((1, d))

                if UPmethod in ['PA', 'PA1', 'PA2', 'PA1_L1', 'PA1_L2', 'PA2_L1', 'PA2_L2',  'OGD', 'OGD_1', 'OGD_2', 'CSOGD_1', 'CSOGD_2', 'CPA', 'CPA1', 'CPA2', 'PA_L1', 'PA_L2', 'PA_I_L1', 'PA_I_L2', 'PA_II_L1','PA_II_L2']:
                    self.C = options.C                     # Regularization parameter
                
                if UPmethod in ['OGD', 'OGD_1', 'OGD_2', 'CSOGD_1', 'CSOGD_2']:
                    self.t = 1                              # Iteration number
                    self.loss_type = options.loss_type      # Loss type
                    self.regularizer = options.regularizer  # Regularization type

                if UPmethod in ['PA1_Csplit', 'PA2_Csplit', 'PA1_CSPLIT', 'PA2_CSPLIT']:
                    self.C = options.C                     # Regularization parameter

                if UPmethod in ['CSRDA_1', 'CSRDA_2']:
                    # Initialize CSRDA-specific parameters
                    # From Algorithm 1, Line 1: "Initialization: w₁ = 0 ∈ R^d, ḡ₀ = 0 ∈ R^d"
                    self.g_bar = np.zeros((1, d))           # Average gradient ḡ_t
                    self.t = 1                               # Timestamp counter (starts at 0, increments to 1 on first call)
                    self.lambda_param = options.lambda_param # λ for sparsity (Section V.C: 10^-1)
                    self.gamma_param = options.gamma_param   # γ for smoothness (Section V.C: 10^-3)
                    self.L = options.L                       # Sliding window length (Section V.C: 100)
                    self.window_labels = []                  # Sliding window for dynamic κ_t estimation
                    # self.kappa_t = options.eta_p / options.eta_n  # Initial κ = ϑ+/ϑ- (Section V.C: 0.95/0.05)
                    self.alpha_t = 1.0                       # Cost parameter ᾱ_t (computed per instance)

                elif UPmethod in ['CSTG_1', 'CSTG_2']:
                    # Initialize CSTG-specific parameters
                    # From Algorithm 1, Line 1: "Initialization: w₁ = 0 ∈ X, g₁ = 0 ∈ X"
                    # Note: g₁ here refers to the gradient accumulator if needed, but CSTG 
                    # mainly uses w updates directly. We initialize w to zeros in the main block.
                    
                    self.t = 1                             # Timestamp counter (starts at 1)
                    
                    # Parameters from CSTG Paper (Table 4)
                    self.g_param = options.g_param         # g: penalty parameter for sparsity
                    self.theta_param = options.theta_param # θ: truncated parameter
                    self.K_param = options.K_param         # K: truncation frequency
                    self.C_param = options.C_param         # C: penalty parameter for cumulative loss
                    self.eta_base = options.eta_base       # η: initial learning rate

        else:
            print(f"Unknown task type or method: {options.task_type}, {options.method}")
            self.C = None  # Default fallback

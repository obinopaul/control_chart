import numpy as np
from regularizers.Regularizer import L0
from regularizers.Regularizer import TGD
from kernels.Kernels import gaussian_kernel
from sklearn import preprocessing as pp
import numpy as np

class Options:

    def __init__(self, method,n,task_type, eta_p=None, eta_n=None):
        # init_options: initialize the options for each method
        #--------------------------------------------------------------------------
        # INPUT:
        #       method:            method name
        #       n:                 number of training instances in the database
        #       task_type:         type of task (bc or mc)
        #       bias:              Add bias weight in the algorithms
        #       regularization     Apply specified regularization in algorithms
        
        self.C = 1  # Default value for C, so it always exists
        self.method = method
        self.t_tick = round(n/15)    #10
        self.task_type = task_type
        self.id_list = np.random.permutation(n)
        self.eta_p = eta_p  # Cost-sensitive parameter for the positive class
        self.eta_n = eta_n  # Cost-sensitive parameter for the negative class
        UPmethod = method.upper()
    
        '''
        
        Initial Parameter values can be modified below
        
        '''
        
        # Options for Binary Classification algorithms
        if (task_type == 'bc'):
            
            if (UPmethod == 'PERCEPTRON' or UPmethod =='PA_L1' or UPmethod =='PA_L2' or UPmethod =='PA' or UPmethod =='CPA'):
                self.bias            = True
                self.p_kernel_degree = 1                            # Input Preprocesing to be applied 
            
            elif (UPmethod == 'GAUSSIAN_KERNEL_PERCEPTRON'):
                self.max_sv       = 100                             # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel                 # Kernel method
                self.sigma        = 1                               # Hyperparameter to use in gaussian_kernel
            
            elif (UPmethod =='PA1' or UPmethod =='PA1_L1' or UPmethod =='PA1_L2' or UPmethod =='PA2' or UPmethod =='PA2_L1'  
                    or UPmethod =='PA2_L2' or UPmethod =='CPA1' or UPmethod =='CPA2' or UPmethod == 'PA_L1' or UPmethod == 'PA_L2' 
                    or UPmethod == 'PA_I_L1' or UPmethod == 'PA_I_L2' or UPmethod == 'PA_II_L1' or UPmethod == 'PA_II_L2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1

            elif (UPmethod == 'PA1_Csplit' or UPmethod == 'PA2_Csplit' or UPmethod == 'PA1_CSPLIT' or UPmethod == 'PA2_CSPLIT'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1
                
            elif (UPmethod =='OGD' or UPmethod == 'OGD_1' or UPmethod == 'OGD_2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.loss_type       = 1                              # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C               = 1
                self.regularizer    = None                            # No regularizer
                #self.regularizer    = L0 (theta = 1.5)               # Coefficient rounding regularizer
                #self.regularizer     = TGD(theta = 1.5, g = 0.025)   # L1 regularizer (gradual decrease of small coefficients)

                
            elif (UPmethod == 'GAUSSIAN_KERNEL_OGD'):
                self.loss_type    = 1                           # type of loss (0, 0-1 loss, 1 - hinge, 2-log, 3-square )
                self.C            = 1
                self.max_sv       = 100                          # Number of instances to keep for kernel approach                 
                self.kernel       = gaussian_kernel              # Kernel method
                self.sigma        = 1                            # Hyperparameter to use in gaussian_kernel
            

            elif (UPmethod == 'CSRDA_1' or UPmethod == 'CSRDA_2'):
                self.bias = True
                self.p_kernel_degree = 1
                # Parameters from paper Section V.C (Experimental Settings)
                self.lambda_param = 0.1        # λ = 10^-1 for sparsity regularization (ℓ1-norm)
                self.gamma_param = 0.001       # γ = 10^-3 for smooth regularization (ℓ2-norm)
                self.L = 100                   # Sliding window length for dynamic imbalance ratio
                # Note: eta_p and eta_n are passed to the algorithm function, not stored in options
                # Paper uses ϑ+ = 0.95 and ϑ- = 0.05 (Section V.C)

            elif (UPmethod == 'CSTG_1' or UPmethod == 'CSTG_2'):
                self.bias = True
                self.p_kernel_degree = 1
                
                # Parameters from CSTG Paper Table 4 (Experimental parameter settings)
                self.g_param = 1.0           # g = 1
                self.theta_param = 1.0       # θ = 1
                self.K_param = 15            # K = 15
                self.C_param = 10.0          # C = 10
                self.eta_base = 0.1          # η = 0.1 (Learning rate)
                
                # Note: Paper uses μ1=0.95 and μ2=0.05, which are passed as eta_p/eta_n inputs

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
            
            else:
                print('Unknown method BC init options.')
        
        # Options for Multiclass Classification algorithms
        elif (task_type == 'mc'):
            
            if (UPmethod == 'M_PERCEPTRONM' or UPmethod == 'M_PERCEPTRONU' or UPmethod == 'M_PERCEPTRONS'):
                self.bias            = True
                self.p_kernel_degree = 1

            elif (UPmethod == 'M_OGD'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C = 1
                self.regularizer = None                           # No regularizer
                #self.regularizer = L0 (theta = 1.5)               # Coefficient rounding regularizer
                #self.regularizer  = TGD(theta = 1.5, g = 0.025)    # L1 regularizer (gradual decrease of small coefficients)

            elif (UPmethod == 'M_PA' or UPmethod == 'M_PA1' or UPmethod =='M_PA2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C = 1

            elif (UPmethod == 'M_CW'):
                self.bias            = True
                self.p_kernel_degree = 1
                self.eta             = 0.75  # in \eta in [0.5,1]
                self.a               = 1

            elif (UPmethod == 'M_SCW1' or UPmethod =='M_SCW2'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.eta             = 0.75
                self.C               = 1
                self.a               = 1

            elif (UPmethod == 'M_AROW'):
                self.bias            = True
                self.p_kernel_degree = 1 
                self.C               = 1      # i.e., parameter r
                self.a               = 1      # default

            elif (UPmethod == 'NEW_ALGORITHM'):
                pass
    
        '''
        
        Hyperparameters tuning ranges can be modified below
        
        '''
        
        self.range_C   = 2**np.arange(-4.0,7.0,1.0)
        self.range_eta = np.arange(0.55,0.95,0.05)
        self.range_b   = np.arange(0.1,0.9,0.1)
        self.range_p   = np.arange(2,10,2) 

        # CSRDA-specific hyperparameter ranges
        self.range_lambda = 10**np.arange(-4.0, 0.0, 0.5)  # λ range: 10^-4 to 10^0
        self.range_gamma = 10**np.arange(-5.0, -1.0, 0.5)  # γ range: 10^-5 to 10^-1
        self.range_L = [50, 100, 150, 200]                 # L range: window sizes

        # CSTG-specific hyperparameter ranges
        self.range_g = 10**np.arange(-2.0, 1.5, 0.5)       # g range: 0.01 to ~31.6
        self.range_K = np.arange(5, 55, 5)                 # K range: 5, 10, ... 50
        self.range_eta_cstg = 10**np.arange(-3.0, 0.5, 0.5)# eta range: 0.001 to ~3.16
        

    def set_csrda_L_range(self, dataset_window_length):
        """
        Dynamically set the range_L based on the dataset's window length.
        
        From CSRDA paper Section V.C: L=100 is used as default.
        We set the range to be relative to the dataset window length to ensure
        reasonable values for hyperparameter tuning.
        
        Args:
            dataset_window_length: The window length w from the dataset (e.g., 10, 20, 45, 100)
        """
        # Set L range based on dataset characteristics
        # Strategy: Use fractions and multiples of dataset window length, 
        # with 100 as reference from paper
        if dataset_window_length <= 20:
            # For small windows, use smaller L values
            self.range_L = [20, 50, 75, 100]
        elif dataset_window_length <= 50:
            # Medium windows
            self.range_L = [50, 75, 100, 150]
        elif dataset_window_length <= 100:
            # Larger windows
            self.range_L = [75, 100, 150, 200]
        else:
            # Very large windows
            self.range_L = [100, 150, 200, 250]
        
        # Always ensure 100 (paper default) is in the range if reasonable
        if 100 not in self.range_L and dataset_window_length <= 150:
            self.range_L.append(100)
            self.range_L.sort()

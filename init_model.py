import numpy as np

class Model:

    def __init__(self, options, d, nb_class):
        # INPUT:
        #  options:     method name and setting
        #        d:     data dimensionality
        # nb_class:     number of class labels

        UPmethod = options.method.upper()
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

        elif options.task_type == 'mc':
            self.task_type = 'mc'
            self.nb_class = nb_class

            # Initialize weight matrix for multiclass tasks
            self.W = np.zeros((int(nb_class), d))

            if UPmethod in ['M_PA1', 'M_PA2', 'M_PA']:
                self.C = options.C

            elif UPmethod in ['M_OGD']:
                self.C = options.C                          # Learning rate parameter
                self.t = 1                                  # Iteration number
                self.regularizer = options.regularizer

            elif UPmethod in ['M_CW', 'M_AROW', 'M_SCW1', 'M_SCW2']:
                self.C = options.C                          # Hyperparameter for algorithms
                # Initialize Sigma matrix for advanced algorithms
                self.Sigma = options.a * np.identity(d)
                
            elif UPmethod == 'NEW_ALGORITHM':
                pass

            else:
                print('Unknown method in multiclass init model.')

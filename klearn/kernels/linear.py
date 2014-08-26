
from klearn.kernels import AbstractKernel
import numpy as np

class Linear(AbstractKernel):
    """
    Linear kernel. This means that we can solve the original
    linear problem but in the gram matrix form.
    """
    def __init__(self):
        pass

    
    def _kernel_function(self, one, many):
        """
        compute the linear dot product
        """
        result = np.dot(many, one.flatten())
        return result

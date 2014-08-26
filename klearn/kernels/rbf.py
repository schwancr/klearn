
from .baseclasses import AbstractKernel
import numpy as np

class RBF(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass raised to the power 'd'. 

    Note that only Vectorized metrics will work here.
    """
    def __init__(self, sigma=1.0):
        
        self.sigma = sigma

    def _kernel_function(self, one, many):
        """
        compute the polynomial inner product between one point and many
        """

        n_points, n_features = many.shape
        one = one.reshape((1, n_features))

        dists2 = np.square(many - one).sum(1)
        result = np.exp(- dists2 / (self.sigma ** 2)).flatten()

        return result

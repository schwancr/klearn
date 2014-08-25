
from .baseclasses import AbstractKernel
import numpy as np

class Gaussian(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass raised to the power 'd'. 

    Note that only Vectorized metrics will work here.
    """

    def __init__(self, metric='l2', sigma=1.0):
        
        self.metric = metric
        self.sigma = sigma

    def _kernel_function(self, one, many):
        """
        compute the polynomial inner product between one point and many
        """

        n_points, n_features = many.shape
        one = one.reshape((1, n_features))

        dists2 = np.square(many - one)
        result = np.exp(- dists2 / (self.sigma ** 2))

        return result

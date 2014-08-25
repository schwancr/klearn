
from .baseclasses import AbstractKernel
import numpy as np

class Polynomial(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass raised to the power 'd'. 

    Note that only Vectorized metrics will work here.
    """

    def __init__(self, degree=2):
        
        self.power = power

    def _kernel_function(self, one, many):
        """
        compute the polynomial inner product between one point and many
        """

        n_points, n_features = many.shape
        one = one.reshape((n_features, 1))

        dots = many.dot(one)
        result = np.power(dots, self.degree)

        return result

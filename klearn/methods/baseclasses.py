from sklearn.base import BaseEstimator
from klearn.kernels import AbstractKernel

class BaseKernelEstimator(BaseEstimator):
    def __init__(self, kernel):
        if not isinstance(kernel, AbstractKernel):
            raise TypeError("kernel must be an instance of klearn.kernels.AbstractKernel")

        self.kernel = kernel

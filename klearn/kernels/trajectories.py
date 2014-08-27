
from klearn.kernels import AbstractKernel
import numpy as np

class TrajectoryKernel(AbstractKernel):
    def __init__(self, kernel):

        if not isinstance(kernel, AbstractKernel):
            raise Exception("kernel must be an instance of klearn.kernels.AbstractKernel")

        self.kernel = kernel


    def _kernel_function(self, one, many):
        """
        compute the kernel function between two trajectories

        Should we center each kernel separately?!
        """

        # one is a numpy array
        # many is a list of numpy arrays

        ni = float(one.shape[0])
        njs = [float(f.shape[0]) for f in many]
        n_trajs = len(many)
        
        result = np.ones(n_trajs)

        for i in xrange(n_trajs):
            result[i] = self.kernel(one, many[i]).sum() / ni / njs[i]

        return result

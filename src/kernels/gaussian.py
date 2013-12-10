
from ktica.kernels import AbstractKernel
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np

class Gaussian(AbstractKernel):
    """
    gaussian kernel with some standard deviation
    """

    def __init__(self, metric, std_dev=1):

        self.metric = metric
        self.std_dev = std_dev
        self.denom = - 2. * std_dev * std_dev

    def __repr__(self):
        return "Gaussian kernel with norm defined by %s" % str(self.metric)

    def prepare_trajectory(self, trajectory):
        return self.metric.prepare_trajectory(trajectory)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return np.exp(np.square(self.metric.one_to_all(np.double(prepared_traj1), np.double(prepared_traj2), index1)) / self.denom)


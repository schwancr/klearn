
from .baseclasses import AbstractKernel
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np

class Polynomial(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass raised to the power 'd'. 

    Note that only Vectorized metrics will work here.
    """

    def __init__(self, metric, power=2):

        if not isinstance(metric, Vectorized):
            raise Exception("Only Vectorized metrics can be used with this kernel.")

        self.metric = metric
        self.power = power

    def __repr__(self):
        return "Polynomial kernel with degree %f and norm defined by %s" % (self.power, str(self.metric))

    def prepare_trajectory(self, trajectory):
        return self.metric.prepare_trajectory(trajectory)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return np.power(1. + np.sum( prepared_traj1[index1:(index1 + 1)] * prepared_traj2, axis=1), self.power)


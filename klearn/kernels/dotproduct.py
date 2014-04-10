
from .baseclasses import AbstractKernel
from msmbuilder.metrics.baseclasses import Vectorized
import numpy as np

class DotProduct(AbstractKernel):
    """
    This kernel is simply the dot product in some vector space given by
    the metric you pass. Note that only Vectorized metrics will work here.

    Also, if you plan on using a kernel trick, this really isn't very 
    useful, as it is the same as working in the vector space defined by
    your metric.
    """

    def __init__(self, metric):

        if not isinstance(metric, Vectorized):
            raise Exception("Only Vectorized metrics can be used with this kernel.")

        self.metric = metric

    def __repr__(self):
        return "NormInduced kernel with norm defined by %s" % str(self.metric)

    def prepare_trajectory(self, trajectory):
        return self.metric.prepare_trajectory(trajectory)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        return np.sum( prepared_traj1[index1:(index1 + 1)] * prepared_traj2, axis=1 ) 


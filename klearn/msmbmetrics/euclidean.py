
from msmbuilder.metrics import Vectorized
import copy

class Euclidean(Vectorized):

    def __init__(self, *args, **kwargs):
        super(Euclidean, self).__init__(*args, **kwargs)

    def prepare_trajectory(self, trajectory):
        if not trajectory.flags.c_contiguous:
            trajectory = copy.copy(trajectory)
        return trajectory

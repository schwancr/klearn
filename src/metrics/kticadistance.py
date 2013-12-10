
from ktica import ktICA
from msmbuilder.metrics import Vectorized
import numpy as np

class ktICADistance(Vectorized):

    def __init__(self, ktica_fn, kernel, num_vecs=None, which_vecs=None,
                 metric='euclidean', p=2):

        super(ktICADistance, self).__init__(metric=metric, p=p)

        self.ktica_obj = ktICA.load(ktica_fn, kernel.one_to_all)

        self.kernel = kernel

        if not which_vecs is None:
            self.which_vecs = np.array(which_vecs).astype(int)

        elif not num_vecs is None:
            self.which_vecs = np.arange(int(num_vecs))
    
        else:
            raise Exception("must input one of which_vecs or num_vecs")


    def prepare_trajectory(self, trajectory):

        ptraj = self.kernel.prepare_trajectory(trajectory)

        return self.ktica_obj.project(ptraj, which=self.which_vecs)

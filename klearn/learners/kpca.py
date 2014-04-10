
import numpy as np
import scipy.linalg
from mdtraj import io
from klearn.learners import BaseLearner, ProjectingMixin

class kPCA(BaseLearner, ProjectingMixin):
    """ 
    class for calculating tICs in a high dimensional feature space
    """

    def __init__(self, kernel):

        """
        Initialize an instance of the ktICA solver

        Paramaters
        ----------
        kernel : klearn.kernels.AbstractKernel instance
            kernel object
        """

        self._Xall = None
        self.K = None
        
        super(self, kPCA).__init__(kernel)


    def add_trining_data(self, trajectory):
        """
        append a trajectory to the calculation. Right now this just appends 
        the trajectory to the concatenated trajectories

        Parameters
        ----------
        trajectory : np.ndarray (2D)
            two dimensional np.ndarray with time in the first axis and 
            features in the second axis

        """

        if self._Xall is None:
            self._Xall = trajectory
        else:
            self._Xall = np.concatenate((self._Xall, trajectory))   
        


    def calculate_matrices(self):
        """
        calculate the two matrices we need, K and Khat and then normalize them
        """

        N = len(self._Xall)

        K = np.zeros((N, N))

        for i in xrange(N):
            K[i] = self.kernel(self._Xall, self._Xall, i)

        # now normalize the matrices.
        self.K = np.zeros((N, N))

        one_N = np.ones((N, N)) / float(N)
        print one_N

        self.K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

        self.K = (self.K + self.K.T) * 0.5


    def solve(self):
        """
        solve the generalized eigenvalue problem for kernel-tICA:
    
        (K Khat^T + Khat^T K) v = w (K K^T + Khat Khat^T)

        Returns
        -------
        eigenvalues : np.ndarray
            eigenvalues from eigensolution
        eigenvectors : np.ndarray
            eigenvectors are stored in the columns
        """

        if self.K is None:
            self.calculate_matrices()

        self.eigen_sol = scipy.linalg.eig(self.K)

        self._sort()

        return self.eigen_sol
    

    def _sort(self):
        """
        sort eigenvectors / eigenvalues so they are decreasing
        """
        if self.eigen_sol is None:
            logger.warning("have not calculated eigenvectors yet...")
            return

        dec_ind = np.argsort(self.eigen_sol[0])[::-1]

        self.eigen_sol = (self.eigen_sol[0][dec_ind], self.eigen_sol[1][:, dec_ind])


    def project(self, trajectory, which):
        """
        project a point onto an eigenvector

        Parameters
        ----------
        trajectory : np.ndarray
            trajectory to project onto eigenvector
        which : list or int
            which eigenvector(s) (0-indexed) to project onto
        
        Returns
        -------
        proj_trajectory : np.ndarray
            projected value of each point in the trajectory
        """

        comp_to_all = []
        
        for i in xrange(len(self._Xall)):
            comp_to_all.append(self.kernel(self._Xall, trajectory, i))
        
        comp_to_all = np.array(comp_to_all).T
        # rows are points from trajectory
        # cols are comparisons to the library points

        if isinstance(which, int):
            which = [which]

        which = np.array(which)

        vecs = self.eigen_sol[1][:, which]

        proj_trajectory = comp_to_all.dot(vecs)

        return proj_trajectory

        

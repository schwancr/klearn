
import numpy as np
import scipy.linalg
from mdtraj import io
import pickle
from klearn.learners import BaseLearner, ProjectingMixin, CrossValidatingMixin
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class kCCA(BaseLearner, ProjectingMixin, CrossValidatingMixin):
    """ 
    class for calculating tICs in a high dimensional feature space
    """

    def __init__(self, kernel, reg_factor=1E-10):

        """
        Initialize an instance of the ktICA solver

        Paramaters
        ----------
        kernel : subclass of kernels.AbstractKernel
            instance of a subclass of kernels.AbstractKernel that defines
            the kernel function for kernel-tICA

        reg_factor : float, optional
            regularization parameter. This class will use a ridge regression
            when solving the generalized eigenvalue problem
        """

        super(kCCA, self).__init__(kernel)

        self.M = None
        
        self.K = None
        self.K_uncentered = None

        self.KK = None
        self.Ka = None

        self.reg_factor = float(reg_factor)

        self._normalized = False

        self.vecs = None
        self.vals = None

        self.vec_vars = None


    def add_training_data(self, X, a):
        """
        append a trajectory to the calculation. Right now this just appends 
        the trajectory to the concatenated trajectories

        Parameters
        ----------
        X : np.ndarray
            two dimensional np.ndarray with time in the first axis and 
            features in the second axis
        X_dt : np.ndarray (2D), optional
            for each point we need a corresponding point that is separated
            in a trajectory by dt. If X_dt is not None, then it should
            be the same length as trajectory such that X[i] comes 
            exactly dt before X_dt[i]. If X_dt is None, then
            we will get all possible pairs from X (X[:-dt, 
            X[dt:]).
        """

        if X.shape[0] != a.shape[0]:
            raise Exception("the data in X should have the same length as the data in a")

        if self.M is None:
            self.M = X
            self.a = a

        else:
            self.M = np.concatenate([self.M, X])
            self.a = np.concatenate([self.a, a])


    def calculate_K(self, precomputed_K=None):
        """
        calculate the gram matrix
        """

        N = len(self.M)

        self.a_mean = self.a.mean()
        self.a_stdev = self.a.std()
        # we need these for later when we predict the a-values

        self.a = (self.a - self.a_mean) / self.a_stdev

        if precomputed_K is None:
            K = np.zeros((N, N))
            for i in xrange(N):
                K[i] = self.kernel.one_to_all(self.M, self.M, i)

        else:
            K = precomputed_K
            if K.shape[0] != N:
                raise Exception("precomputed K is the wrong size for this data.")

        # now normalize the matrices.
        self.K = np.zeros((N, N))

        one_N = np.ones((N, N)) / float(N)

        self.K_uncentered = np.array(K)

        self.K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

        self.K = (self.K + self.K.T) * 0.5

        self.KK = self.K.dot(self.K)
        self.Ka = self.K.dot(self.a)


    def solve(self):
        """
        solve the generalized eigenvalue problem for kernel-CCA:

        Ka.T aK \beta = \lambda (K K + \eta I) \beta

        Parameters
        ----------
        num_pcs : int, optional
            in order to solve the tICA problem we have to do it in PCA 
            space, otherwise the RHS becomes singular.

        Returns
        -------
        eigenvalues : np.ndarray
            eigenvalues from eigensolution
        eigenvectors : np.ndarray
            eigenvectors are stored in the columns
        """

        if self.K is None:
            self.calculate_matrices()

        N = self.K.shape[0]
        
        lhs = np.outer(self.Ka, self.Ka)
        rhs = self.KK + np.eye(K.shape[0]) * self.regularization_strength

        self.vals, self.vecs = scipy.linalg.eig(lhs, b=rhs)

        if np.abs(self.vals).max() > 1:
            logger.warn("some eigenvalues are not bounded by one. "
                        "You might try changing the regularization factor.")


        if np.abs(self.vals.imag).max() > 1E-12:
            logger.warn("some eigenvalues are not real. You might"
                        " try chaniging the regularization factor.")

        else:
            self.vals = self.vals.real
            self.vecs = self.vecs.real

        self._normalize()
        self._sort()

        return self.vals, self.vecs
    

    def _sort(self):
        """
        sort eigenvectors / eigenvalues so they are decreasing
        """
        if self.vals is None:
            logger.warn("have not calculated eigenvectors yet...")
            return

        if not self._normalized:
            self._normalize()

        dec_ind = np.argsort(self.vals.real)[::-1]
        self.vals = self.vals[dec_ind]
        self.vecs = self.vecs[:, dec_ind]


    def _normalize(self):
        """
        normalize the eigenvectors to unit variance.

        Note: This is the same normalization as the right eigenvectors
            of the transfer operator under the assumption that self._Xall 
            is distributed according to the true equlibrium populations.
        """

        M = float(self.K.shape[0])

        vKK = self.vecs.T.dot(KK)

        self.vec_vars = np.sum(vKK * self.vecs.T, axis=1) / M
        # dividing by M instead of M - 1. Shouldn't really matter...

        norm_vecs = self.vecs / np.sqrt(self.vec_vars)

        self.vecs = norm_vecs


    def predict(self, X, which):
        """
        project a point onto an eigenvector

        Parameters
        ----------
        X : np.ndarray
            data to project onto eigenvector
        which : array_like or int
            which eigenvector(s) (0-indexed) to project onto
        
        Returns
        -------
        pred_a : np.ndarray
            projected value of each point in the trajectory
        """

        comp_to_all = []

        for i in xrange(len(self.M)):
            comp_to_all.append(self.kernel.one_to_all(self.M, X, i))
        
        comp_to_all = np.array(comp_to_all).T
        # rows are points from trajectory
        # cols are comparisons to the library points

        M = self.K_uncentered.shape[0]

        comp_to_all = comp_to_all - np.reshape(comp_to_all.sum(axis=1), (-1, 1)) / float(M) \
                        - self.K_uncentered.sum(axis=0) / float(M) \
                        + self.K_uncentered.sum() / float(M) / float(M)

        if isinstance(which, int):
            which = [which]

        which = np.array(which)

        if not self._normalized:
            self._normalize()

        vecs = self.vecs[:, which]

        pred_a = comp_to_all.dot(vecs)
        pred_a = pred_a * self.a_stdev + self.a_mean

        return pred_a


    def project(self, X, which):
        return self.predict(X, which)


    def evaluate(self, a_pred, a_actual, callback_a=lambda a : a):
        a0 = callback_a(a_actual)
        a1 = callback_a(a_pred)

        return np.sum(np.abs(a0 - a1))
        

    def save(self, output_fn):
        """
        save results to a .h5 file
        """
    
        kernel_str = pickle.dumps(self.kernel)

        io.saveh(output_fn, vals=self.vals,
            vecs=self.vecs, K=self.K, 
            K_uncentered=self.K_uncentered, 
            reg_factor=np.array([self.reg_factor]),
            M=self.M, a=self.a, a_mean=self.a_mean,
            a_stdev=self.a_stdev, 
            kernel_str=np.array([kernel_str]))


    @classmethod
    def load(cls, input_fn, kernel=None):
        """
        load a ktica object saved via the .save method. 

        Parameters
        ----------
        input_fn : str
            input filename
        kernel : kernel instance, optional
            kernel to use when calculating inner products. If None,
            then we will look in the file. If it's not there, then an 
            exception will be raised

        Returns
        -------
        kt : ktica instance
        """

        f = io.loadh(input_fn)

        if not kernel is None:
            kernel = kernel
        elif 'kernel_str' in f.keys():
            kernel = pickle.loads(f['kernel_str'][0])
        else:
            raise Exception("kernel_str not found in %s. Need to pass a kernel object")

        kt = cls(kernel, reg_factor=f['reg_factor'][0])  
        # dt and reg_factor were saved as arrays with one element

        kt.K_uncentered = f['K_uncentered']
        kt.K = f['K']

        kt.M = f['M'].astype(np.double)
        kt.a = f['a']
        kt.a_mean = f['a_mean']
        kt.a_stdev = f['a_stdev']


        kt.vals = f['vals']
        kt.vecs = f['vecs']

        kt._normalized = False
        kt._sort()
        # ^^^ sorting also normalizes 
     
        return kt



import numpy as np
import scipy.linalg
from mdtraj import io
import pickle
from klearn.learners import BaseLearner, ProjectingMixin, CrossValidatingMixin
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ktICA(BaseLearner, ProjectingMixin, CrossValidatingMixin):
    """ 
    class for calculating tICs in a high dimensional feature space
    """

    def __init__(self, kernel, dt, reg_factor=1E-10):

        """
        Initialize an instance of the ktICA solver

        Paramaters
        ----------
        kernel : subclass of kernels.AbstractKernel
            instance of a subclass of kernels.AbstractKernel that defines
            the kernel function for kernel-tICA

        dt : int
            correlation lagtime to compute the tICs for

        reg_factor : float, optional
            regularization parameter. This class will use a ridge regression
            when solving the generalized eigenvalue problem
        """

        super(ktICA, self).__init__(kernel)

        self._Xa = None
        self._Xb = None
        
        self.K = None
        self.K_uncentered = None

        self.reg_factor = float(reg_factor)

        self.dt = int(dt)

        self._normalized = False

        self.vecs = None
        self.vals = None

        self.vec_vars = None


    def add_training_data(self, X, X_dt=None):
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

        if X_dt is None:
            A = X[:-self.dt]
            B = X[self.dt:]

        else:
            if X_dt.shape != X.shape:
                raise Exception("X and X_dt should be same shape!")

            A = X
            B = X_dt

        if self._Xa is None:
            self._Xa = A
            self._Xb = B
        else:
            self._Xa = np.concatenate((self._Xa, A)) 
            self._Xb = np.concatenate((self._Xb, B))  


    def calculate_matrices(self, precomputed_K=None):
        """
        calculate the two matrices we need, K and Khat and then normalize them
        """

        N = len(self._Xa) * 2

        self._Xall = np.concatenate((self._Xa, self._Xb))

        if precomputed_K is None:
            K = np.zeros((N, N))
            for i in xrange(N):
                K[i] = self.kernel.one_to_all(self._Xall, self._Xall, i)

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


    def solve(self, num_pcs=None):
        """
        solve the generalized eigenvalue problem for kernel-tICA:
    
        (K Khat^T + Khat^T K) v = w (K K^T + Khat Khat^T)

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

        N = self.K.shape[0] / 2

        rot_mat = np.zeros((2 * N, 2 * N))
        rot_mat[:N, N:] = np.eye(N)
        rot_mat += rot_mat.T

        lhs = self.K.dot(rot_mat).dot(self.K)
        rhs = self.K.dot(self.K) + np.eye(self.K.shape[0]) * self.reg_factor

        eigen_sol = scipy.linalg.eig(lhs, b=rhs)

        self.vals = eigen_sol[0]
        self.vecs = eigen_sol[1]

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


    def _normalize(self, use_regularization=False):
        """
        normalize the eigenvectors to unit variance.

        Note: This is the same normalization as the right eigenvectors
            of the transfer operator under the assumption that self._Xall 
            is distributed according to the true equlibrium populations.
        """

        KK = self.K.dot(self.K)

        M = float(self.K.shape[0])

        vKK = self.vecs.T.dot(KK)

        self.vec_vars = np.sum(vKK * self.vecs.T, axis=1) / M
        # dividing by M instead of M - 1. Shouldn't really matter...
        if use_regularization:
            self.vec_vars += self.reg_factor * np.square(self.vecs).sum(0) / M

        norm_vecs = self.vecs / np.sqrt(self.vec_vars)

        self.vecs = norm_vecs


    def project(self, X, which):
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
        proj_X : np.ndarray
            projected value of each point in the trajectory
        """

        comp_to_all = []

        for i in xrange(len(self._Xall)):
            comp_to_all.append(self.kernel.one_to_all(self._Xall, X, i))
        
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

        proj_X = comp_to_all.dot(vecs)

        return proj_X


    def evaluate(self, *args, **kwargs):
        return self.evaluate_eigenvalues(*args, **kwargs)

    
    def evaluate_eigenvalues(self, X=None, X_dt=None, proj_X=None, proj_X_dt=None,
        num_vecs=10, scheme='all', timestep=1):
        """
        Evaluate the solutions based on new data. For each ktIC, this function 
        computes the autocorrelation value. Then these eigenvalues can be
        combined via a number of schemes to return a "score"

        Parameters
        ----------
        X : np.ndarray
            data with features in the second axis
        X_dt : np.ndarray
            data such that X_dt[i] is observed one timestep after X[i]. Note
            that this timestep need not be the same as the dt specified for
            solving the ktICA solution
        proj_X : np.ndarray
            same as X, but already projected using self.project(X, ...)
        proj_X_dt : np.ndarray
            same as X_dt, but already projected using self.project(X_dt, ...)
        num_vecs : int, optional
            number of vectors to evaluate the likelihood at
        scheme : one of {'l2', 'l1', 'sum', 'all'}, optional
            the scheme for evaluating the quality of a model:
            - 'l2' : l2 norm on the deviation from the training set eigenvalues
            - 'l1' : l1 norm on the deviation from the training set eigenvalues
            - 'sum' : sum of the test set eigenvalues (this scheme essentially
                doesn't care if the test eigenvalues are too slow, but does care
                if they are too fast.)
            - 'all' : return all of the above

        Returns
        -------
        score : float or np.ndarray
            the score of this projection (or three scores if scheme=='all')
        """
        
        if proj_X is None or proj_X_dt is None:
            proj_X = self.project(X, which=np.arange(num_vecs))
            proj_X_dt = self.project(X_dt, which=np.arange(num_vecs))

        if proj_X.shape != proj_X_dt.shape:
            raise Exception("pairs of trajectory data should have the same shape")

        if timestep < self.dt:
            raise Exception("can't model dynamics less than original dt.")

        elif timestep == self.dt:
            exponent = 1

        else:
            if (timestep % self.dt):
                raise Exception("for timestep > dt, timestep must be a multiple of dt.")

            exponent = int(round(timestep / self.dt))

        scheme = scheme.lower()
        if not scheme in ['l1', 'l2', 'sum', 'rel', 'all']:
            raise Exception("scheme (%s) must be one of 'l1', 'l2', 'sum', or 'all'" % scheme)
    
        proj_X = proj_X# - proj_X.mean(0)
        proj_X_dt = proj_X_dt# - proj_X_dt.mean(0)

        numerator = np.sum(proj_X * proj_X_dt, axis=0) / len(proj_X)

#        denominator = np.sum(proj_X * proj_X, axis=0) + np.sum(proj_X_dt * proj_X_dt, axis=0)
#        denominator /= 2.0
        denominator = 1.0

        ratio = numerator / denominator

        vals = np.power(self.vals[:num_vecs], exponent)

        l1 = np.abs(ratio - vals).sum()
        l2 = np.square(ratio - vals).sum()
        s = np.sum(ratio)
        rel = np.abs((ratio - vals) / vals).sum()

        print "l1: %.2e - l2: %.2e - sum: %.2e - rel: %.2e" % (l1, l2, s, rel)

        if scheme == 'l1':
            score = np.abs(ratio - vals).sum()

        elif scheme == 'l2':
            score = np.square(ratio - vals).sum()

        elif scheme == 'sum':
            score = np.sum(ratio)

        elif scheme == 'rel':
            score = rel

        elif scheme == 'all':
            score = np.array([l1, l2, s, rel])

        return score


    def log_likelihood(self, equil, equil_dt, X=None, X_dt=None, proj_X=None,
        proj_X_dt=None, num_vecs=10, timestep=1, min_prob=0.0):
        """
        Evaluate the solutions based on new data X. This uses the assumption that
        the ktICA solutions are the eigenfunctions of the Transfer operator.
        This means we can decompose the transfer operator into a sum of terms
        along each ktICA solution, which gives a probability of a new trajectory.

        Using Bayes' rule this can be translated into the likelihood of the
        ktICA solution.

        Parameters
        ----------
        equil : np.ndarray
            equilibrium probability of each conformation in the data, X
        equil_dt : np.ndarray
            equilibrium probability of each conformation in the data, X_dt
        X : np.ndarray
            data with features in the second axis
        X_dt : np.ndarray
            data such that X_dt[i] is observed one timestep after X[i]. Note
            that this timestep need not be the same as the dt specified for
            solving the ktICA solution
        proj_X : np.ndarray
            same as X, but already projected using self.project(X, ...)
        proj_X_dt : np.ndarray
            same as X_dt, but already projected using self.project(X_dt, ...)
        num_vecs : int, optional
            number of vectors to evaluate the likelihood at
        timestep : time separating X from X_dt (should be in the same units
            as self.dt

        Returns
        -------
        log_like : float
            log likelihood of the new data given the ktICA solutions
        """

        if len(equil.shape) == 1:
            equil = np.reshape(equil, (-1, 1))
        

        if len(equil_dt.shape) == 1:
            equil_dt = np.reshape(equil_dt, (-1, 1))
        

        if proj_X is None or proj_X_dt is None:
            proj_X = self.project(X, which=np.arange(num_vecs))
            proj_X_dt = self.project(X_dt, which=np.arange(num_vecs))


        if np.unique([len(ary) for ary in [proj_X, proj_X_dt, equil, equil_dt]]).shape[0] != 1:
            raise Exception("X, X_dt, equil, and equil_dt should all be the same length.")


        if timestep < self.dt:
            raise Exception("can't model dynamics less than original dt.")

        elif timestep == self.dt:
            exponent = 1

        else:
            if (timestep % self.dt):
                raise Exception("for timestep > dt, timestep must be a multiple of dt.")

            exponent = int(round(timestep / self.dt))

        N = proj_X.shape[0]

        # the first right eigenvector sends everything to unity
        proj_X = np.hstack([np.ones((N, 1)), proj_X])
        proj_X_dt = np.hstack([np.ones((N, 1)), proj_X_dt])

        vals = np.concatenate([[1], self.vals[:num_vecs]]).real
        vals = np.power(vals, exponent)
        vals = np.reshape(vals, (-1, 1))

        temp_array = proj_X * proj_X_dt * equil_dt
        # don't multiply by muA because that is the normalization
        # constraint on the output PDF
        temp_array = temp_array.dot(vals)
        # NOTE: The above likelihood is merely proportional to the actual likelihood
        # we would really need to multiply by a volume of phase space, since this
        # is the likelihood PDF...
        #temp_array[np.where(temp_array <=0)] = temp_array[np.where(temp_array > 0)].min() / 1E6
        bad_ind = np.where(temp_array <= 0)[0]
        if len(bad_ind) > 0:
            logger.warn("%d (out of %d) pairs were assigned nonpositive probabilities" % (len(bad_ind), len(temp_array)))

        if min_prob > 0:
            temp_array[bad_ind] = min_prob

        log_like = np.log(temp_array).sum()

        return log_like


    def save(self, output_fn):
        """
        save results to a .h5 file
        """
    
        kernel_str = pickle.dumps(self.kernel)

        io.saveh(output_fn, vals=self.vals,
            vecs=self.vecs, K=self.K, 
            K_uncentered=self.K_uncentered, 
            reg_factor=np.array([self.reg_factor]),
            traj=self._Xall, dt=np.array([self.dt]),
            normalized=np.array([self._normalized]), 
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

        kt = cls(kernel, f['dt'][0], reg_factor=f['reg_factor'][0])  
        # dt and reg_factor were saved as arrays with one element

        kt.K_uncentered = f['K_uncentered']
        kt.K = f['K']

        kt._Xall = f['traj'].astype(np.double)
        kt._Xa = kt._Xall[:len(kt._Xall) / 2]
        kt._Xb = kt._Xall[len(kt._Xall) / 2:]


        kt.vals = f['vals']
        kt.vecs = f['vecs']

        kt._normalized = False
        kt._sort()
        # ^^^ sorting also normalizes 
     
        return kt


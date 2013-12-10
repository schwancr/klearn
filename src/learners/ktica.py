
import numpy as np
import scipy.linalg
from msmbuilder import io
import pickle
from klearn.kernels import AbstractKernel, ProjectingMixin, CrossValidatingMixin

class ktICA(AbstractKernel, ProjectingMixin, CrossValidatingMixin):
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

        super(self, ktICA).__init__(kernel)

        self._Xa = None
        self._Xb = None
        
        self.K = None
        self.K_uncentered = None

        self.reg_factor = float(reg_factor)

        self.dt = int(dt)

        self._normalized = False
        self.acf_vals = None
        self.vec_vars = None


    def add_training_data(self, trajectory, trajectory_dt=None, prepped=True):
        """
        append a trajectory to the calculation. Right now this just appends 
        the trajectory to the concatenated trajectories

        Parameters
        ----------
        trajectory : np.ndarray (2D)
            two dimensional np.ndarray with time in the first axis and 
            features in the second axis
        trajectory_dt : np.ndarray (2D), optional
            for each point we need a corresponding point that is separated
            in a trajectory by dt. If trajectory_dt is not None, then it should
            be the same length as trajectory such that trajectory[i] comes 
            exactly dt before trajectory_dt[i]. If trajectory_dt is None, then
            we will get all possible pairs from trajectory (trajectory[:-dt, 
            trajectory[dt:]).
        prepped : 
        
        """

        if trajectory_dt is None:
            A = trajectory[:-self.dt]
            B = trajectory[self.dt:]

        else:
            if trajectory_dt.shape != trajectory.shape:
                raise Exception("trajectory and trajectory_dt should be same shape!")

            A = trajectory
            B = trajectory_dt

        if self._Xa is None:
            self._Xa = A
            self._Xb = B
        else:
            self._Xa = np.concatenate((self._Xa, A)) 
            self._Xb = np.concatenate((self._Xb, B))  


    def calculate_matrices(self):
        """
        calculate the two matrices we need, K and Khat and then normalize them
        """

        N = len(self._Xa) * 2

        K = np.zeros((N, N))

        self._Xall = np.concatenate((self._Xa, self._Xb))

        for i in xrange(N):
            K[i] = self.kernel.one_to_all(self._Xall, self._Xall, i)

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

        lhs = K.dot(rot_mat).dot(K)
        rhs = K.dot(K) + np.eye(K.shape[0]) * self.reg_factor

        self.eigen_sol = scipy.linalg.eig(lhs, b=rhs)

        self._normalize()

        self._sort()

        return self.eigen_sol
    

    def _sort(self):
        """
        sort eigenvectors / eigenvalues so they are decreasing
        """
        if self.eigen_sol is None:
            logger.warn("have not calculated eigenvectors yet...")
            return

        if not self._normalized:
            self._normalize()

        vecs = self.eigen_sol[1]
        term2 = self.reg_factor * np.square(vecs).sum(axis=0) / vecs.shape[0]

        self.acf_vals = self.eigen_sol[0].real * (1 + term2.real)

        dec_ind = np.argsort(self.acf_vals)[::-1]
        # sort them in descending order

        good_acf_bools = np.abs(self.acf_vals[dec_ind] <= 1)
        good_acf_inds = dec_ind[np.where(good_acf_bools)]
        bad_acf_inds = dec_ind[np.where(1 - good_acf_bools)][::-1]

        # don't throw anything out, but put the "bad" ones at the end sorted
        # by least "badness"
        end_sorted = np.concatenate([good_acf_inds, bad_acf_inds])

        self.acf_vals = self.acf_vals[end_sorted]
        self.eigen_sol = (self.eigen_sol[0][end_sorted], self.eigen_sol[1][:, end_sorted])


    def _normalize(self):
        """
        normalize the eigenvectors to unit variance.

        Note: This is the same normalization as the right eigenvectors
            of the transfer operator under the assumption that self._Xall 
            is distributed according to the true equlibrium populations.
        """

        KK = self.K.dot(self.K)

        M = float(self.K.shape[0])

        vKK = self.eigen_sol[1].T.dot(KK)

        self.vec_vars = np.sum(vKK * self.eigen_sol[1].T, axis=1) / M
        # dividing by M instead of M - 1. Shouldn't really matter...

        norm_vecs = self.eigen_sol[1] / np.sqrt(self.vec_vars)

        self.eigen_sol = (self.eigen_sol[0], norm_vecs)

        self._normalized = True


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
            comp_to_all.append(self.kernel.one_to_all(self._Xall, trajectory, i))
        
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

        vecs = self.eigen_sol[1][:, which]

        proj_trajectory = comp_to_all.dot(vecs)

        return proj_trajectory


    def evaluate(self, equilA, equilB, trajA=None, trajB=None, projA=None,
        projB=None, num_vecs=10, timestep=1):

        if not trajA is None and not trajB is None:
            if np.unique([len(ary) for ary in [trajA, trajB, equilA, equilB]]).shape[0] != 1:
                raise Exception("trajA, trajB, equilA, and equilB should all be the same length.")
        else:
            if np.unique([len(ary) for ary in [projA, projB, equilA, equilB]]).shape[0] != 1:
                raise Exception("trajA, trajB, equilA, and equilB should all be the same length.")

        if timestep < self.dt:
            raise Exception("can't model dynamics less than original dt.")

        elif timestep == self.dt:
            exponent = 1

        else:
            if (timestep % self.dt):
                raise Exception("for timestep > dt, timestep must be a multiple of dt.")

            exponent = timestep / self.dt

        if projA is None:
            projA = self.project(trajA, which=np.arange(num_vecs))
    
        if projB is None:
            projB = self.project(trajB, which=np.arange(num_vecs))

        N = projA.shape[0]
        projA = np.hstack([np.ones((N, 1)), projA])
        projB = np.hstack([np.ones((N, 1)), projB])
        vals = np.concatenate([[1], self.acf_vals[:num_vecs]]).real
        vals = np.power(vals, exponent)
        vals = np.reshape(vals, (-1, 1))
        if len(equilA.shape) == 1:
            equilA = np.reshape(equilA, (-1, 1))
        if len(equilB.shape) == 1:
            equilB = np.reshape(equilB, (-1, 1))
        temp_array = projA * projB * equilB
        # don't multiply by muA because that is the normalization
        # constraint on the output PDF
        temp_array = temp_array.dot(vals)
        # NOTE: The above likelihood is merely proportional to the actual likelihood
        # we would really need to multiply by a volume of phase space, since this
        # is the likelihood PDF...
        log_like = np.log(temp_array).sum()
        print log_like
        return log_like


    def save(self, output_fn):
        """
        save results to a .h5 file
        """
    
        kernel_str = pickle.dumps(self.kernel)

        io.saveh(output_fn, ktica_vals=self.eigen_sol[0],
            ktica_vecs=self.eigen_sol[1], K=self.K, 
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


        kt.eigen_sol = (f['ktica_vals'], f['ktica_vecs'])

        kt._normalized = False
        kt._sort()
        # ^^^ sorting also normalizes 
     
        return kt


import abc
from klearn.kernels import AbstractKernel

class BaseLearner(object):
    """
    Basic learner that implements what we want from the kernel 
    learners
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, kernel):
        
        if not isinstance(kernel, AbstractKernel):
            raise Exception("kernel must be of type AbstractKernel")

        self.kernel = kernel


    @abc.abstractmethod
    def add_training_data(self, X):
        """
        Add new training data to the learner. It is up to the
        subclass to decide what to do with the data, as it may
        be advantageous to calculate something on the fly, or 
        just store it until later.
        """

        raise NotImplementedError("not implemented!")

    
    @abc.abstractmethod
    def solve(self):
        """
        Solve the learning method given all of the training 
        data. 
        """
        
        raise NotImplementedError("not implemented!")


class CrossValidatingMixin(object):
    """
    mixin for learners that can be cross-validated
    """
    
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, X):
        """
        cross-validate new data given the solution
        """

        raise NotImplementedError("not implemented!")


class ProjectingMixin(object):
    """
    mixin for learners that project new data onto some
    solutions. I.e. these learners will transform the 
    data into one or more dimensions
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def project(self, X, num_vecs=10):
        """
        project data, X, onto some number of solutions
        """

        raise NotImplementedError("not implemented!")
    

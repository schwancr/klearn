import numpy as np
from klearn.methods import kRidgeRegression
from klearn.kernels import Linear
from sklearn.metrics import r2_score
from matplotlib.pyplot import *
def test_krr_regP():
    
    dim = 5
    n = 1000
    ntest = 1001    
    eta = 10

    pref = np.random.random(size=dim) - 0.5

    #pref /= np.sqrt(pref.dot(pref))

    Xtrain = np.random.random((n, dim)) + 1.0
    ytrain = Xtrain.dot(pref) + np.random.normal(scale=0.05, size=n)

    Xtest = np.random.random((ntest, dim)) + 1.0
    yref = Xtest.dot(pref)

    krr = kRidgeRegression(kernel=Linear(), eta=eta)
    krr.fit(Xtrain, ytrain)

    ytest = krr.transform(Xtest).flatten()

    print ytrain.mean()
    assert krr.score(Xtest, yref) > 0.98

    scatter(Xtrain.dot(pref), ytrain, color='blue', alpha=0.5)
    scatter(yref, ytest, color='red', alpha=0.5)
    show()

def test_krr_regbeta():
    
    dim = 5
    n = 1000
    ntest = 1001    
    eta = 10

    pref = np.random.random(size=dim) - 0.5

    #pref /= np.sqrt(pref.dot(pref))

    Xtrain = np.random.random((n, dim))
    ytrain = Xtrain.dot(pref) + np.random.normal(scale=0.05, size=n)

    Xtest = np.random.random((ntest, dim)) 
    yref = Xtest.dot(pref)

    krr = kRidgeRegression(kernel=Linear(), eta=eta, regularize_beta=True)
    krr.fit(Xtrain, ytrain)

    ytest = krr.transform(Xtest).flatten()

    assert krr.score(Xtest, yref) > 0.98

    #scatter(Xtrain.dot(pref), ytrain, color='blue', alpha=0.5)
    #scatter(yref, ytest, color='red', alpha=0.5)
    #show()

if __name__ == '__main__':
    test_krr_regP()
    test_krr_regbeta()

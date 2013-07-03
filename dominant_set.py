import numpy as np
from numpy.linalg import norm

def dominant_set(A, x=None, epsilon=1.0e-4):
    """Compute the dominant set of the similarity matrix A with the
    replicator dynamics optimization approach. Convergence is reached
    when x changes less than epsilon.

    See: 'Dominant Sets and Pairwise Clustering', by Massimiliano
    Pavan and Marcello Pelillo, PAMI 2007.
    """
    if x is None:
        x = np.ones(A.shape[0])/float(A.shape[0])
        
    distance = epsilon*2
    while distance > epsilon:
        x_old = x.copy()
        # x = x * np.dot(A, x) # this works only for dense A
        x = x * A.dot(x) # this works both for dense and sparse A
        x = x / x.sum()
        distance = norm(x - x_old)
        print x.size, distance

    return x


if __name__=="__main__":

    from sklearn.metrics import pairwise_distances
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    np.random.seed(1)

    n = 1000
    d = 2

    X, y = make_blobs(n, d, centers=3)

    D = pairwise_distances(X, metric='sqeuclidean')

    sigma2 = np.median(D)
    
    S = np.exp(-D / sigma2)

    x = dominant_set(S, epsilon=2e-4)
    
    if d==2:
        plt.figure()
        for yi in np.unique(y):
            plt.plot(X[y==yi,0], X[y==yi,1], 'o')

        plt.title('Dataset')

    plt.figure()
    plt.imshow(S, interpolation='nearest')
    plt.title('similarity matrix')

    idx = np.argsort(x)[::-1]
    B = S[idx,:][:,idx]
    plt.figure()
    plt.imshow(B, interpolation='nearest')
    plt.title('Re-arranged similarity matrix')
    plt.figure()
    plt.semilogy(np.sort(x))
    plt.title('Sorted weighted characteristic vector (x)')

    cutoff = np.median(x[x>0])
    print "cutoff:", cutoff
    plt.figure()
    plt.plot(X[x<=cutoff,0], X[x<=cutoff,1], 'bo')
    plt.plot(X[x>cutoff,0], X[x>cutoff,1], 'ro')
    plt.title("Dominant set")
    
    plt.show()

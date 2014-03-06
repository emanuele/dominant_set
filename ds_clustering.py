"""Dominant set clustering: iteratively find the dominant set and then
remove it from the dataset.
"""

import numpy as np
from dominant_set import dominant_set

if __name__ == '__main__':

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

    if d==2:
        plt.figure()
        for yi in np.unique(y):
            plt.plot(X[y==yi,0], X[y==yi,1], 'o')

        plt.title('Dataset')


    while S.size > 10:
        x = dominant_set(S, epsilon=2e-4)
        cutoff = np.median(x[x>0])

        plt.figure()
        plt.plot(X[x<=cutoff,0], X[x<=cutoff,1], 'bo')
        plt.plot(X[x>cutoff,0], X[x>cutoff,1], 'ro')
        plt.title("Dominant set")

        # remove the dominant set
        idx = x <= cutoff
        S = S[idx, :][:, idx]
        X = X[idx, :]
        
    plt.show()
        
        

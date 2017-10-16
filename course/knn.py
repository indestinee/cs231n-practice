import numpy as np

def kth_largest(x, k):
    return np.sort(x)[k - 1]

# Nearest Neighbor Brute Force
class nn_bf(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        n = x.shape[0]
        y = np.zeros(n, dtype=self.y.dtype)
        
        for i, data in enumerate(x):
            distances = np.sqrt(np.sum((data - self.x) ** 2, axis = 1))
            min_index = np.argmin(distances)
            y[i] = self.y[min_index]
        return y

class knn_bf(object):
    def __init__(self, x):
        self.cnt = np.zeros(x, dtype=int)

    def train(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x, k = 1):
        y = np.zeros(x.shape[0], dtype=self.y.dtype)
        for i, data in enumerate(x):
            self.cnt[:] = 0
            distances = np.sqrt(np.sum((data - self.x) ** 2, axis = 1))
            p = np.where(distances <= kth_largest(distances, k))
            self.cnt[p] += 1
            min_index = self.y[np.argmax(self.cnt)]
            y[i] = min_index
        return y



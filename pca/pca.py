import numpy as np
import matplotlib.pyplot as plt

class PCA(object):
    def __init__(self, n, data):
        self.n = n
        self.data = data

    def zero_mean(self):
        mean = np.mean(self.data, axis=0)
        self.data -= mean

    def fit(self):
        self.zero_mean()
        cov = np.cov(self.data, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(cov))
        indices = np.argsort(eigVals)[-self.n:]
        n_eigVects = eigVects[:, indices]
        print(n_eigVects)
        new_data = data  * n_eigVects
        return new_data


def load_data():
    sigma = np.array([[10, 5], [5, 5]])
    return np.random.multivariate_normal([0, 0], sigma, 1000)

if __name__ == "__main__":
    data = load_data()
    model = PCA(1, data)
    new_data = model.fit()
    plt.plot(data[:, 0], data[:, 1], 'r.')
    plt.plot(np.ravel(new_data), np.zeros(1000), 'b+')
    plt.show()
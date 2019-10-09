import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
TRAIN_SAMPLES = 500
PREDICT_SAMPLES = 200

class KNN(object):
    def __init__(self, k=1):
        self.k = k

    def fit(self, x, y, predicted_x):
        dist_mat = np.empty((predicted_x.shape[1], x.shape[1]))
        for col in range(predicted_x.shape[1]):
            dist_mat[col, :] = self.dist(x, predicted_x[:, col])
        index = np.argsort(dist_mat, axis=1)[:, :self.k]
        return [1 if np.count_nonzero(row) > self.k//2 else 0 for row in y[index]]

    def dist(self, a, b):
        b = np.reshape(b, (b.shape[0], 1))
        return np.sum(np.square(a - b), axis=0)
    
def load_train_data():
    x1 = np.random.randn(2, TRAIN_SAMPLES) - 0.5
    x2 = np.random.randn(2, TRAIN_SAMPLES) + 0.5
    x = np.hstack((x1, x2))
    y = np.hstack((np.zeros(TRAIN_SAMPLES), np.ones(TRAIN_SAMPLES)))
    return x, y

def load_test_data():
    x1 = np.random.randn(2, TRAIN_SAMPLES) - 1
    x2 = np.random.randn(2, TRAIN_SAMPLES) + 1
    x = np.hstack((x1, x2))
    return x

if __name__ == "__main__":
    x_train, y_train = load_train_data()
    plt.scatter(x_train[0], x_train[1], c=y_train)
    plt.show()

    x_test = load_test_data()
    model = KNN(1)
    predicted_y = model.fit(x_train, y_train, x_test)
    plt.scatter(x_test[0], x_test[1], c=predicted_y)
    plt.show()
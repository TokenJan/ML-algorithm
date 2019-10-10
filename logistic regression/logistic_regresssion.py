import numpy as np
import matplotlib.pyplot as plt

TRAIN_SAMPLES = 500
PREDICT_SAMPLES = 500

class LogisticRegression(object):
    # 初始化逻辑回归模型参数
    def __init__(self, learning_rate=0.001, n_iterations=20):
        self.w = np.random.randn(3)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    # Sigmoid函数，S型曲线
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def fit(self, x, y):
        x = np.insert(x, 2, 1, axis=0)
        x_test = load_test_data()
        x_test = np.insert(x_test, 2, 1, axis=0)

        for _ in range(self.n_iterations):
            # evaluate model every iteration    
            y_predict = self.predict(x_test)
            plt.scatter(x_test[0], x_test[1], c=y_predict)
            plt.show()

            # train the model
            h_x = x.T.dot(self.w)
            predicted_y = self.sigmoid(h_x)
            w_grad = x.dot(predicted_y - y)
            self.w = self.w - self.learning_rate * w_grad


    def predict(self, x):
        h_x = x.T.dot(self.w)
        p = self.sigmoid(h_x)
        return np.round(p)


def load_train_data():
    x1 = np.random.uniform(-2, 2, TRAIN_SAMPLES)
    x2 = np.random.uniform(-2, 2, TRAIN_SAMPLES)
    x = np.array([x1, x2])
    y = [0 if x1_ - x2_ < 2  else 1 for x1_, x2_ in zip(x1, x2)]
    return x, y

def load_test_data():
    x1 = np.random.uniform(-2, 2, PREDICT_SAMPLES)
    x2 = np.random.uniform(-2, 2, PREDICT_SAMPLES)
    x = np.array([x1, x2])
    return x

if __name__ == "__main__":
    x_train, y_train = load_train_data()
    plt.scatter(x_train[0], x_train[1], c=y_train)
    plt.show()

    model = LogisticRegression()
    model.fit(x_train, y_train)
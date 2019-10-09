import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self):
        self.predicted_w = 0
        self.predicted_b = 0

    def least_square(self, x, y):
        x_mean = np.mean(x)
        self.predicted_w = np.sum(y * (x - x_mean)) / (np.sum(x**2) - np.sum(x)**2 / len(x))
        self.predicted_b = np.sum(y - self.predicted_w * x) / len(x)

def load_data():
    w = 5
    b = 2
    x = np.array(range(100))
    noise = np.random.randn(100) * 10
    y = w * x + b + noise
    return x, y

if __name__ == "__main__":
    model = LinearRegression()
    x, y = load_data()
    model.least_square(x, y)
    predicted_y = model.predicted_w * x + model.predicted_b
    plt.plot(x, y)
    plt.plot(x, predicted_y)
    plt.show()
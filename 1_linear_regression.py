import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        pass

    @staticmethod
    def load_data():
        ds = load_diabetes()
        X, Y = shuffle(ds.data, ds.target, random_state=42)
        X = X.astype(np.float32)
        Y = Y.reshape((-1, 1))
        return X, Y

    @staticmethod
    def init_params(dims):
        W = np.random.rand(dims, 1)
        b = 0
        return W, b

    @staticmethod
    def loss_grad(X, Y, W, b):
        """Calculate linear regression MSE loss, weight and bias gradients."""
        # number of samples
        num_samples = X.shape[0]
        # predicted Y
        Y_hat = np.dot(X, W) + b
        # loss / example
        loss = np.sum((Y - Y_hat) ** 2, axis=None) / num_samples
        # gradients / example
        dW = np.dot(X.T, (Y_hat - Y)) / num_samples
        db = np.sum(Y_hat - Y) / num_samples

        return loss, dW, db

    def train(self, X, Y, lr, epochs):
        W, b = self.init_params(X.shape[1])
        loss_list = []
        for epoch in range(epochs):
            loss, dW, db = self.loss_grad(X, Y, W, b)
            loss_list.append(loss)
            W -= lr * dW
            b -= lr * db
            if epoch % 10000 == 0:
                print(f'Epoch {epoch:6d} loss {loss:.4f}')
        return loss_list, W, b

    @staticmethod
    def predict(X, W, b):
        return np.dot(X, W) + b


if __name__ == '__main__':
    lr = LinearRegression()
    X, Y = lr.load_data()
    loss_list, W, b = lr.train(X, Y, 1e-2, int(1e5))
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

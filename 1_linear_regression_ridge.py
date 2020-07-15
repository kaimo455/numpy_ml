import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


class LinearRegressionRidge:
    def __init__(self, l2):
        self.W = None
        self.b = None
        self.l2 = l2

    @staticmethod
    def load_data():
        ds = load_diabetes()
        X, Y = shuffle(ds.data, ds.target, random_state=42)
        X = X.astype(np.float32)
        Y = Y.reshape((-1, 1))
        return X, Y

    def init_params(self, dims):
        self.W = np.random.rand(dims, 1)
        self.b = 0

    def loss_grad(self, X, Y):
        """Calculate linear regression MSE loss, weight and bias gradients."""
        num_samples = X.shape[0]
        Y_hat = np.dot(X, self.W) + self.b
        loss = np.sum((Y - Y_hat) ** 2, axis=None) / num_samples + self.l2 * np.sum(self.W ** 2, axis=None)
        dW = np.dot(X.T, (Y_hat - Y)) / num_samples + self.l2 * 2 * self.W
        db = np.sum(Y_hat - Y, axis=None) / num_samples
        return loss, dW, db

    def train(self, X, Y, lr, epochs):
        # init weights and bias
        self.init_params(X.shape[1])
        loss_list = []
        for epoch in range(epochs):
            loss, dW, db = self.loss_grad(X, Y)
            loss_list.append(loss)
            # update weight and bias
            self.W -= lr * dW
            self.b -= lr * db
            if epoch % 10000 == 0:
                print(f'Epoch {epoch:4d} loss {loss:.4f}')
        return loss_list


if __name__ == '__main__':
    X, Y = LinearRegressionRidge.load_data()
    lr_ridge = LinearRegressionRidge(0.1)
    loss_list = lr_ridge.train(X, Y, 1e-2, int(1e5))
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    print(lr_ridge.W)

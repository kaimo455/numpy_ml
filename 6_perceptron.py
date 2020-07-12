import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class Perceptron:
    def __init__(self):
        self.W = None
        self.b = None

    def _init_params(self, dim):
        self.W = np.random.randn(dim)
        self.b = 0

    def linear(self, x):
        """
        Calculate linear transformation of a sample, of shape [dim]
        :param x: a sample instance
        :type x: numpy array
        :return: numpy array of single value
        :rtype: float
        """
        return np.dot(x, self.W) + self.b

    def train(self, X, Y, lr):
        # init weight and bias
        self._init_params(X.shape[1])
        # while loop to train wight and bias
        has_error = True
        while has_error:
            has_error = False
            # iter each sample instance
            for x, y in zip(X, Y):
                # misclassified point
                if y * self.linear(x) <= 0:
                    # update weight and bias
                    self.W += lr * y * x
                    self.b += lr * y
                    # set has_error to True
                    has_error = True

    def predict(self, X):
        return ((np.dot(X, self.W) + self.b) > 0).astype(np.int8)


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # only take first two-columns and target label as our training data
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, Y = data[:, :-1], data[:, -1]
    Y = np.where(Y == 1, np.ones_like(Y), np.ones_like(Y) * -1)
    # plot perceptron
    perceptron = Perceptron()
    perceptron.train(X, Y, 0.01)
    _x_min, _x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    _y_min, _y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(_x_min, _x_max, 0.02), np.arange(_y_min, _y_max, 0.02))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    # plot data
    plt.scatter(df['sepal length'][:50], df['sepal width'][:50], c='yellow', label='class 0')
    plt.scatter(df['sepal length'][50:100], df['sepal width'][50:100], c='black', label='class 1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

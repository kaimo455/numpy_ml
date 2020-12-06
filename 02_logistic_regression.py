import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        pass

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def init_params(dims):
        W = np.random.rand(dims, 1)
        b = 0
        return W, b

    def loss_grad(self, X, Y, W, b):
        """Calculate logistic regression cross-entropy loss, weight and bias gradients."""
        # number of samples
        num_samples = X.shape[0]
        # predicted Y probability
        Y_hat = self.sigmoid(np.dot(X, W) + b)
        # loss
        loss = (-1) / num_samples * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat), axis=None)
        # weight and bias gradients
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
            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d} loss {loss:.4f}')

        return loss_list, W, b

    def predict_prob(self, X, W, b):
        return self.sigmoid(np.dot(X, W) + b)

    def predict(self, X, W, b):
        prob = self.predict_prob(X, W, b)
        return (prob > 0.5).astype(np.int8)

    @staticmethod
    def load_data():
        X, Y = make_classification(n_samples=1000, n_features=30, n_informative=2, n_redundant=2, n_repeated=10,
                                   n_classes=2, n_clusters_per_class=2, random_state=42)
        Y = Y.reshape(-1, 1)
        return X, Y

if __name__ == '__main__':
    model = LogisticRegression()
    X, Y = model.load_data()
    loss_list, W, b = model.train(X, Y, 1e-3, int(1e5))
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

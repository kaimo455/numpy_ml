import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class KNearestNeighbor:

    def __init__(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def _compute_dist(self, X):
        """Calculate distances between test points and training points.

        for one sample pair, use matrix to calculate:
            dist = np.sqrt(np.sum((x_i - x_j)**2, axis=-1))
        for one test sample and all training samples pair, use matrix to calculate:
            dist = np.sqrt(np.sum((X_train - x_j)**2, axis=-1)).reshape(1, -1)
        for all train-test sample pairs, use matrix to calculate:
            dist = np.sum(X_test**2, axis=-1)[:, np.newaxis] + np.sum(X_train**2, axis=-1)[:, np.newaxis].T - 2*X_test*X_train.T
        """

        dists = np.sum(X ** 2, axis=-1).reshape(-1, 1) + np.sum(self.X_train ** 2, axis=-1).reshape(1, -1) \
                - 2 * np.dot(X, self.X_train.T)
        return dists

    def predict(self, X, k=1):
        dists = self._compute_dist(X)
        indices = np.argsort(dists, axis=-1)[:, :k]
        num_test, num_train = X.shape[0], self.Y_train.shape[0]
        Y_closest = np.take_along_axis(np.broadcast_to(self.Y_train, (num_test, num_train)), 
                                       indices,
                                       axis=-1)
        Y_pred = np.apply_along_axis(lambda arr: Counter(arr).most_common(1)[0][0], -1, Y_closest)
        return Y_pred

    @staticmethod
    def create_data():
        data = load_iris()
        return shuffle(data.data, data.target)


if __name__ == '__main__':
    X, Y = KNearestNeighbor.create_data()
    knn = KNearestNeighbor(X, Y)
    Y_perd = knn.predict(X, k=5)
    print(accuracy_score(Y, Y_perd))

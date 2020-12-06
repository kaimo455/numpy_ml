import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class WeakClassifier:
    """A single decision tree stump as weak classifier."""

    def __init__(self):
        self.feature_index = None
        self.feature_value = None
        # according to feat_ind and feat_val to classify those samples into `self.cls` class
        self.cls = None
        # weak clf weight
        self.alpha = None


class Adaboost:
    def __init__(self, n_estimators):
        self.estimators = None
        self.n_estimators = n_estimators

    def fit(self, X, y):
        num_sample, num_feature = X.shape

        # init weights
        W = np.full(num_sample, 1 / num_sample)
        self.estimators = []

        # for each estimator
        for _ in range(self.n_estimators):
            min_error = float('inf')
            weak_clf = WeakClassifier()

            # greedy finding best split
            for feature_idx in range(num_feature):
                unique_values = np.unique(X[:, feature_idx], axis=None)
                for val in unique_values:

                    pred_cls = 1
                    y_hat = np.ones_like(y)
                    # larger than val, predicted as 1, other wise -1
                    y_hat[X[:, feature_idx] <= val] *= - 1
                    # calculate error rate
                    error = np.sum(W[y_hat != y])
                    # flip prediction if error rate is larger than 0.5
                    if error > 0.5:
                        error = 1 - error
                        pred_cls = -1

                    # if find a better split point, then update weak clf
                    if error < min_error:
                        weak_clf.cls = pred_cls
                        weak_clf.feature_index = feature_idx
                        weak_clf.feature_value = val
                        min_error = error

            # calculate current weak clf weight
            weak_clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-7))
            # update data distribution
            y_hat = np.full_like(y, weak_clf.cls)
            y_hat[X[:, weak_clf.feature_index] <= weak_clf.feature_value] *= -1
            W *= np.exp(-weak_clf.alpha * y * y_hat)
            W /= np.sum(W, axis=None)

            self.estimators.append(weak_clf)

    def predict(self, X):
        num_samples = X.shape[0]
        y_hat = np.zeros((num_samples,))
        # iterate each weak clf
        for clf in self.estimators:
            _y_hat = np.full((num_samples,), clf.cls)
            _y_hat[X[:, clf.feature_index] <= clf.feature_value] *= -1
            y_hat += clf.alpha * _y_hat

        return np.sign(y_hat)


if __name__ == '__main__':
    data = load_digits()
    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(data.target == digit1)[0], np.where(data.target == digit2)[0])
    y = data.target[idx]
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    clf = Adaboost(n_estimators=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'accuracy: {np.mean(y_test == y_pred, axis=None):.4f}')

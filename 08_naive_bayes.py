import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self._classes = None  # unique values of class
        self._Py_prior = None  # class prior probability
        self._Pxy_prior = None  # conditional prior probability
        self._fitted = False  # whether the model is fitted

    def fit(self, X, Y):
        assert (len(X.shape) == 2)
        assert (len(Y.shape) == 1)
        # calculate Py_prior dict (cls -> prob)
        self._Py_prior = (Y.value_counts() / Y.shape[0]).to_dict()
        # calculate classes
        self._classes = self._Py_prior.keys()
        # calculate Pxy_prior dict ((col, col_val, cls) -> prob)
        self._Pxy_prior = {}
        for cls in self._classes:  # for each class
            _sub_X = X[Y == cls]
            for col in X.columns:  # for each feature dimension
                for col_val, prob in (_sub_X[col].value_counts() / _sub_X.shape[0]).to_dict().items():
                    self._Pxy_prior[(col, col_val, cls)] = prob
        self._fitted = True

    def predict(self, X):
        pred_prob = np.ones((X.shape[0], len(self._classes)))
        for cls_idx, cls in enumerate(self._classes):  # for each class
            for col in X.columns:  # for each feature
                for sample_idx, col_val in enumerate(X[col]):  # for each sample
                    pred_prob[sample_idx, cls_idx] *= self._Pxy_prior.get((col, col_val, cls), 0)
            pred_prob[:, cls_idx] *= self._Py_prior[cls]
        return np.array(list(self._classes))[np.argmax(pred_prob, axis=1)]


if __name__ == '__main__':
    X = pd.DataFrame({'x0': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                      'x1': ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']})
    Y = pd.Series([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    nb = NaiveBayes()
    nb.fit(X, Y)
    pred = nb.predict(pd.DataFrame({'x0': [1, 2, 3, 2], 'x1': ['S', 'M', 'L', 'S']}))
    print(pred)

import numpy as np
from sklearn.datasets import load_iris, load_boston
from tqdm import tqdm

"""
General Decision Tree.
"""


class DecisionTree:
    """Super class of regression DecisionTree and classification DecisionTree."""

    class TreeNode:
        """Tree Node."""

        def __init__(self, is_leaf, column_index=None, column_value=None, left_child=None, right_child=None,
                     score=None):
            """
            Tree node as leaf by default.
            :param score: output score of this leaf
            :type score: float
            """
            self.is_leaf = is_leaf
            self.score = score
            self.column_index = column_index
            self.column_value = column_value
            self.left_child = left_child
            self.right_child = right_child

    def __init__(self, min_samples_split=2, min_gain=1e-7, max_depth=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth if (max_depth or max_depth == 0) else float('inf')
        # member methods to be override in sub-class
        self.cal_leaf_value = None
        self.split_metric = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, curr_depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, curr_depth: int) -> TreeNode:
        """
        Recursively build decision tree, every pass of this function will create/return a TreeNode.
        :param X: feature matrix of shape [n_sample, n_feature]
        :type X: Numpy Array
        :param y: target label of shape [n_sample]
        :type y: Numpy Array
        :param curr_depth: tree depth of the current pass of this function
        :type curr_depth: int
        :return: tree node either a split node or leaf node
        :rtype: TreeNode
        """
        n_sample, n_feature = X.shape
        max_gain = 0.0
        split_feature = None
        split_value = None
        split_left_index = None
        split_right_index = None

        # if current_depth < max_depth and number of sample in X > min_samples_split, then find the best split
        if curr_depth < self.max_depth and n_sample >= self.min_samples_split:
            for feature_index in range(n_feature):  # Iterate each feature and find the best split
                unique_values = np.unique(X[:, feature_index])
                for feature_val in sorted(unique_values):  # Exact greedy algorithm
                    left_index = X[:, feature_index] <= feature_val
                    right_index = ~left_index
                    # calculate the split metric w.r.t. this split, e.g. information gain / variance reduction
                    if self.split_metric:
                        gain = self.split_metric(y, [y[left_index], y[right_index]])
                    else:
                        raise NotImplementedError(f'split metric not defined.')
                    # if find a better gain, update max_gain and other settings
                    if gain > max_gain:
                        max_gain = gain
                        split_feature = feature_index
                        split_value = feature_val
                        split_left_index = left_index
                        split_right_index = right_index

        # if cannot find a effective split, i.e. max_gain == 0 or max_gain < min_gain, then return a leaf node.
        if max_gain == 0 or max_gain < self.min_gain:
            # TODO: add loss function value in node attribute.
            return self.TreeNode(is_leaf=True, score=self.cal_leaf_value(y) if self.cal_leaf_value else None)
        else:
            left_node = self._build_tree(X[split_left_index], y[split_left_index], curr_depth + 1)
            right_node = self._build_tree(X[split_right_index], y[split_right_index], curr_depth + 1)
            return self.TreeNode(is_leaf=False, column_index=split_feature, column_value=split_value,
                                 left_child=left_node, right_child=right_node)

    def predict(self, X):
        """
        Predict input samples X.
        :param X: input samples matrix of shape [n_sample, n_feature]
        :type X: Numpy.ndarray
        :return: prediction of shape [n_sample, ]
        :rtype: Numpy.ndarray
        """
        return np.vectorize(self._predict, signature='(n)->()')(X)

    def _predict(self, x):
        """
        Make prediction of a single sample.
        :param x: single input sample of shape [n_feature]
        :type x: Numpy ndarray
        :return: prediction value
        :rtype: float
        """
        node = self.root
        while not node.is_leaf:
            if x[node.column_index] <= node.column_value:
                node = node.left_child
            else:
                node = node.right_child
        return node.score


"""
Regression & Classification Decision Tree.
"""


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, **kwargs):
        super(DecisionTreeRegressor, self).__init__(**kwargs)
        self.cal_leaf_value = predict_mean
        self.split_metric = variance_reduction

    def fit(self, X, y):
        super(DecisionTreeRegressor, self).fit(X, y)


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, **kwargs):
        super(DecisionTreeClassifier, self).__init__(**kwargs)
        self.cal_leaf_value = predict_voting
        self.split_metric = entropy_gain

    def fit(self, X, y):
        super(DecisionTreeClassifier, self).fit(X, y)


"""
General Gradient Boosting Decision Tree(GBDT).
"""


class GBDT:
    """Super class of regression GBDT and classification GBDT."""

    # TODO: implement the classification version

    def __init__(self, n_estimators=100, learning_rate=1e-2, max_depth=None, min_samples_split=2, min_gain=1e-7,
                 row_subsample=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.row_subsample = row_subsample if (row_subsample and (0 < row_subsample < 1)) else 1
        self.estimators = []
        # use regression tree to fix both value(regression)/probability(classification)
        self.BaseEstimator = DecisionTreeRegressor
        # member methods to be override in sub-class
        self.loss_obj = None

    def fit(self, X, y):
        # first estimator
        base_estimator = self.BaseEstimator(max_depth=0, min_samples_split=self.min_samples_split,
                                            min_gain=self.min_gain)
        base_estimator.fit(X, y)
        self.estimators.append(base_estimator)
        y_pred = base_estimator.predict(X)
        # rest estimators
        for i in tqdm(range(1, self.n_estimators)):
            base_estimator = self.BaseEstimator(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                                min_gain=self.min_gain)
            sample_idx = np.random.choice(X.shape[0], int(self.row_subsample * X.shape[0]), replace=False)
            base_estimator.fit(X[sample_idx], -self.loss_obj.gradient(y_true=y, y_pred=y_pred)[sample_idx])
            self.estimators.append(base_estimator)
            y_pred += self.learning_rate * base_estimator.predict(X)

    def predict(self, X):
        pred_y = self.estimators[0].predict(X)
        for estimator in self.estimators[1:]:
            pred_y += self.learning_rate * estimator.predict(X)
        return pred_y


"""
Regression GBDT.
"""


class GBDTRegressor(GBDT):
    def __init__(self, **kwargs):
        super(GBDTRegressor, self).__init__(**kwargs)
        self.loss_obj = SquareLoss()


"""
Loss objects for GBDT.
"""


class SquareLoss:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return 0.5 * (y_true - y_pred) ** 2

    @staticmethod
    def gradient(y_true, y_pred):
        return - (y_true - y_pred)


"""
Decision Tree node split algorithms.

- entropy gain
- variance reduction
"""


def entropy(array: np.ndarray):
    """
    Calculate entropy.
    :param array: a array contains category values
    :type array: Numpy ndarray
    :return: entropy
    :rtype: float
    """
    _, counts = np.unique(array, return_counts=True)
    probs = counts / array.shape[0]
    return - np.sum(probs * np.log2(probs), axis=None)


def entropy_gain(total_array: np.ndarray, split_arrays: list):
    """
    Calculate entropy gain/difference between before and after.
    :param total_array: array contains categorical values before split
    :type total_array: Numpy ndarray
    :param split_arrays: list of arrays which is sub set of total_array
    :type split_arrays: list
    :return: entropy gain
    :rtype: float
    """
    entropy_D = entropy(total_array)
    entropy_DA = 0
    for split_arr in split_arrays:
        weight = split_arr.shape[0] / total_array.shape[0]
        entropy_DA += weight * entropy(split_arr)
    return entropy_D - entropy_DA


def variance(array: np.ndarray):
    """
    Calculate variance.
    :param array: a array contains category values
    :type array: Numpy ndarray
    :return: variance
    :rtype: float
    """
    # if the input array is empty, then the variance is zero
    if len(array) == 0:
        return 0
    return np.sum((array - array.mean()) ** 2, axis=0) / array.shape[0]


def variance_reduction(total_array: np.ndarray, split_arrays: list):
    """
    Calculate variance reduction between before and after.
    :param total_array: array contains categorical values before split
    :type total_array: Numpy ndarray
    :param split_arrays: list of arrays which is sub set of total_array
    :type split_arrays: list
    :return: variance reduction
    :rtype: float
    """
    variance_D = variance(total_array)
    variance_DA = 0
    for split_arr in split_arrays:
        weight = split_arr.shape[0] / total_array.shape[0]
        variance_DA += weight * variance(split_arr)
    return variance_D - variance_DA


"""
Decision Tree leaf score (prediction) calculation algorithms.

- hard classification: majority voting
- soft classification: each class probabilities
- regression: mean
- regression: objective function oriented (XGBoost)
"""


def predict_voting(array: np.ndarray):
    """
    Majority voting.
    :param array: predicted target labels
    :type array: Numpy ndarray
    :return: class label
    :rtype: Union[str, float, int]
    """
    classes, counts = np.unique(array, return_counts=True)
    return classes[np.argmax(counts)]


def predict_proba(array: np.ndarray):
    raise NotImplementedError()


def predict_mean(array: np.ndarray):
    """
    Take mean value of predicted target labels
    :param array: predicted target labels
    :type array: Numpy ndarray
    :return: mean value of predicted target labels
    :rtype: float
    """
    return np.mean(array, axis=None)


def predict_obj():
    raise NotImplementedError()


"""
Main entrance.
"""

if __name__ == '__main__':
    # classification decision tree
    data = load_iris()
    X, y = data.data, data.target
    dt_clf = DecisionTreeClassifier(max_depth=2)
    dt_clf.fit(X, y)
    print(f'Iris dataset(classification):\n\tAccuracy: {np.mean(dt_clf.predict(X) == y):.4f}')
    # regression decision tree
    data = load_boston()
    X, y = data.data, data.target
    dt_reg = DecisionTreeRegressor(max_depth=4)
    dt_reg.fit(X, y)
    print(f'Boston dataset(regression):\n\tMSE: {np.sum(np.square(dt_reg.predict(X) - y), axis=None) / y.shape[0]:.4f}')
    # regression GBDT
    gbdt_reg = GBDTRegressor(n_estimators=10, learning_rate=1, max_depth=4)
    gbdt_reg.fit(X, y)
    print(
        f'Boston dataset(regression):\n\tMSE: {np.sum(np.square(gbdt_reg.predict(X) - y), axis=None) / y.shape[0]:.4f}')

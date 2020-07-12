import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


class ID3Tree:
    class Node:
        def __init__(self, col_name):
            self.name = col_name
            self.children = {}

        def connect(self, col_val, node):
            self.children[col_val] = node

    def __init__(self, data: pd.DataFrame, label: str):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node('root')

    @staticmethod
    def _entropy(array):
        """
        Calculate entropy.
        :param array: a array or list contains category values
        :type array: array of strings
        :return: entropy value
        :rtype: float
        """
        array = list(array)
        unique_values = set(array)
        probs = [array.count(val) / len(array) for val in unique_values]
        return - np.sum(probs * np.log2(probs), axis=None)

    @staticmethod
    def _split_data(data, column):
        """
        Split DataFrame base on different value on column.
        :param data: data DataFrame
        :type data: DataFrame
        :param column: column split on
        :type column: string
        :return: a string-DataFrame pair dict
        :rtype: dict
        """
        if column not in data.columns:
            raise ValueError(f'{column} not found in data.')
        unique_values = data[column].unique()
        split_data = {val: data[data[column] == val] for val in unique_values}
        return split_data

    def find_split(self, data, label):
        """
        Find the best split based on information gain.
        :param data: data DataFrame
        :type data: DataFrame
        :param label: target label
        :type label: Union[int, float, string]
        :return: max_information_gain, best_column, best_split_data
        :rtype: float, string, dict
        """
        entropy_D = self._entropy(data[label])
        max_information_gain = float('-inf')
        best_column = None
        best_split_data = None
        # iterate each column
        for col in data.columns.drop(label):
            # split on current column
            split_data = self._split_data(data, col)
            # calculate information gain
            entropy_DA = 0.0
            for col_val, sub_data in split_data.items():
                entropy_Di = self._entropy(sub_data[label])
                entropy_DA += len(sub_data) / len(data) * entropy_Di
            information_gain = entropy_D - entropy_DA
            # compare with max_information_gain
            if information_gain > max_information_gain:
                best_column = col
                max_information_gain = information_gain
                best_split_data = split_data
        return max_information_gain, best_column, best_split_data

    def show_tree(self, node, tabs):
        print(tabs + node.name)
        for col_label, child in node.children.items():
            print(tabs + '\t(' + str(col_label) + ')')
            self.show_tree(child, tabs + '\t\t')

    def create_tree(self):
        self._create_tree(self.root, '', self.data, self.columns)

    def _create_tree(self, parent_node, parent_col_label, data, columns):
        max_information_gain, best_column, best_split_data = self.find_split(data[columns], self.label)
        # break recursive
        if not best_column:
            return
        node = self.Node(best_column)
        parent_node.connect(parent_col_label, node)
        # recursively create decision tree
        for col_label, sub_data in best_split_data.items():
            self._create_tree(node, col_label, sub_data, columns.drop(best_column))


if __name__ == '__main__':
    data = load_wine()
    label = 'target'
    data = pd.DataFrame(np.concatenate([data.data, data.target.reshape(-1, 1)], axis=1),
                        columns=data.feature_names + [label])
    tree_id3 = ID3Tree(data, label)
    tree_id3.create_tree()
    tree_id3.show_tree(tree_id3.root, '')

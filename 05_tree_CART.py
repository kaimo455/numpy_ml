import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


class CARTTree:
    class Node:
        def __init__(self, col_name):
            self.name = col_name
            self.children = {}

        def connect(self, col_val, node):
            self.children[col_val] = node

    def __init__(self, df: pd.DataFrame, target_label: str):
        self.df = df
        self.target_label = target_label
        self.root = self.Node('root')
        self.original_cols = self.df.columns

    @staticmethod
    def _gini(array):
        """
        Calculate Gini index for array.
        :param array: a array or list contains category values
        :type array: array of strings
        :return: gini index value
        :rtype: float
        """
        array = list(array)
        unique_values = set(array)
        probs = [array.count(val) / len(array) for val in unique_values]
        return 1 - np.sum(np.array(probs) ** 2, axis=None)

    def create_tree(self):
        self._create_tree(self.root, '', self.df, self.original_cols)

    def _create_tree(self, parent_node, parent_col_val, df, columns):
        max_gini_gain, best_col, best_split_df = self.find_split(df[columns])
        # break recursive if no best_col found
        if best_col is None:
            return
        node = self.Node(best_col)
        parent_node.connect(parent_col_val, node)
        # recursively create decision tree
        for col_val, sub_df in best_split_df.items():
            self._create_tree(node, col_val, sub_df, columns.drop(best_col))

    def find_split(self, df):
        """
        Find the best split based on gini index gain.
        :param df: data DataFrame
        :type df: DataFrame
        :return: max_gini_gain, best_col, best_split_df
        :rtype: float, string, dict
        """
        # calculate the gini index before split
        gini_D = self._gini(df[self.target_label].tolist())

        # initialize gini_gain, best_col, best_split_df
        max_gini_gain = float('-inf')
        best_col = None
        best_split_df = None

        # iter each columns in df to find the best split column
        for col in df.columns.drop(self.target_label):  # split on current col
            split_df = self._split_data(df, col)  # calculate gini index after split
            gini_DA = 0.0
            for col_val, sub_df in split_df.items():
                gini_Di = self._gini(sub_df[self.target_label].tolist())  # calculate gini index for each sub_data
                gini_DA += len(sub_df) / len(df) * gini_Di  # sum up gini_Di to gini_DA
            gini_gain = gini_D - gini_DA  # calculate gini index gain
            if gini_gain > max_gini_gain:
                best_col = col
                best_split_df = split_df
                max_gini_gain = gini_gain

        return max_gini_gain, best_col, best_split_df

    @staticmethod
    def _split_data(df, col):
        # TODO: CART is binary tree, the _split_data() method is wrong. It should be:
        #   categorical feature -> create True/False branches
        #   continuous feature -> create less/larger than branches
        """
        Split DataFrame base on different value on column.
        :param df: data DataFrame
        :type df: pandas DataFrame
        :param col: column split on
        :type col: string
        :return: dict of key is label_val - val is sub_df
        :rtype: dict
        """
        if col not in df.columns:
            raise ValueError(f'{col} not found in data.')
        col_vals = df[col].unique()
        split_data = {col_val: df[df[col] == col_val] for col_val in col_vals}
        return split_data

    def show_tree(self, node, tabs):
        print(tabs + node.name)
        for col_val, child in node.children.items():
            print(tabs + '\t(' + str(col_val) + ')')
            self.show_tree(child, tabs + '\t\t')


if __name__ == '__main__':
    data = load_wine()
    label = 'target'
    data = pd.DataFrame(np.concatenate([data.data, data.target.reshape(-1, 1)], axis=1),
                        columns=data.feature_names + [label])
    tree_cart = CARTTree(data, label)
    tree_cart.create_tree()
    tree_cart.show_tree(tree_cart.root, '')

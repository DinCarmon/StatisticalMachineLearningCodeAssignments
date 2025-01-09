# Introduction to Machine Learning course - Code Assignment 1
# Authors: Tal Grossman, Amir Sharif Jamal, Din Carmon:
# Tal Grossman      201512282
# Amir Sharif Jamal 213850811
# Din Carmon        209325026

# Part 2: Decision Trees

import numpy as np
from typing import Callable, List, Tuple

def compute_entropy(p : int):
    """
    p - some probability value between 0 and 1.
    return the entropy of a Bernoulli distribution with probability p.
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

class Split:
    def __init__(self, feature_idx, threshold):
        self.feature_idx = feature_idx
        self.threshold = threshold

    def __call__(self, X):
        """
        Compute the label of the split function on X.
        """
        return X[self.feature_idx] < self.threshold

    def do_split(self, s : List):
        """
        s - a set of samples, each sample is a tuple (X), where
        X is a numpy array of features / (X, label).
        """
        s_0 = []
        s_1 = []


        for i in range(len(s)):
            if type(s[i]) == tuple:
                x = s[i][0]
            else:
                x = s[i]

            if self(x):
                s_1.append(s[i])
            else:
                s_0.append(s[i])
        return s_0, s_1

    def __repr__(self):
        return f"x[{self.feature_idx}] < {self.threshold}?"

def build_all_possible_splits(s : List[np.ndarray]):
    """
        s - a subset of examples, each example is a tuple (X), where
        X is a numpy array of features
    """
    splits = []

    for index in range(s[0].shape[0]):
        possible_values = [sample[index] for sample in s]
        for threshold in np.unique(possible_values):
            sp = Split(index, threshold)
            splits.append(sp)
    return splits

class TreeNode:
    def __init__(self, value):
        self.is_leaf = True
        self.value = value  # The value of the node
        self.children = []  # List to store child nodes

    def add_child(self, child_node):
        """
        Adds a child node to the current node.
        Assume first added child is the left child,
        second added child is the right child.
        """
        self.children.append(child_node)
        self.is_leaf = False

    def __call__(self, X):
        """
        Compute the label of X based on the tree.
        """

        # First check if it is a leaf
        if type(self.value) == int:
            return self.value

        if self.value(X):
            return self.children[1](X)
        else:
            return self.children[0](X)

    def __repr__(self, level=0):
        """String representation of the tree for visualization."""
        ret = " " * (2 * level) + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

def compute_positive_ratio(s : List[Tuple[np.ndarray, int]]):
    labels = np.array([sample[1] for sample in s])
    counts = np.bincount(labels.ravel().astype(int))

    # Check if there are only 0 labeled samples:
    if len(counts) == 1:
        return 0
    else:
        return counts[1] / counts.sum()

def majority_vote(s : List[Tuple[np.ndarray, int]]):
    if compute_positive_ratio(s) >= 0.5:
        return 1
    else:
        return 0

def build_subtree(root_node : TreeNode,
                  s : List[Tuple[np.ndarray, int]],
                  a : List[Split]):
    """
    The function builds a decision tree using the ID3 algorithm.

    s - a subset of examples, each example is a tuple (X, y), where
        X is a numpy array of features, and y is a label (0 or 1)
    a - Possible split functions.
    """
    if len(s) == 0:
        raise ValueError("s must be a non-empty list")
    if not root_node.is_leaf:
        raise ValueError("root_node must be a leaf node on call")

    # If all labels are c return c
    labels = [sample[1] for sample in s]
    if np.std(labels) == 0:
        root_node.value = int(s[0][1])
        return

    # If A is empty, return leaf d where d is majority label in S
    if len(a) == 0:
        root_node.value = int(majority_vote(s))

    # Find best split
    best_split = None
    best_split_potential_reduction = -np.inf
    for split in a:
        s_0, s_1 = split.do_split(s)
        f = len(s_1) / len(s) # The fraction of samples reaching the right subtree

        if len(s_0) == 0 or len(s_1) == 0:
            potential_reduction = 0
        else:
            q_l = compute_positive_ratio(s_0)
            q_r = compute_positive_ratio(s_1)
            potential_before = compute_entropy(((1 - f) * q_l) + \
                                               (f * q_r))
            potential_after = ((1 - f) * compute_entropy(q_l)) + \
                              (f * compute_entropy(q_r))
            potential_reduction = potential_before - potential_after

        if potential_reduction > best_split_potential_reduction:
            best_split = split
            best_split_potential_reduction = potential_reduction

    # Do the split
    s_0, s_1 = best_split.do_split(s)

    a_without_best_split = [split for split in a if split != best_split]
    m = majority_vote(s)

    # compute left subtree recursively
    for s_i in [s_0, s_1]:
        if len(s_i) > 0:
            # We shall compute the value, i.e. the split name of the child in the future recursive calls
            child_tree = TreeNode(None)
            build_subtree(child_tree, s_i, a_without_best_split)
            root_node.add_child(child_tree)
        else:
            child_tree = TreeNode(m)
            root_node.add_child(child_tree)

    root_node.value = best_split

def build_tree(s : List[Tuple[np.ndarray, int]],
               a : List[Split]):
    """
    The function builds a decision tree using the ID3 algorithm.

    s - a subset of examples, each example is a tuple (X, y), where
        X is a numpy array of features, and y is a label (0 or 1)
    a - Possible split functions.
    """
    root_node = TreeNode(None)
    build_subtree(root_node, s, a)
    return root_node
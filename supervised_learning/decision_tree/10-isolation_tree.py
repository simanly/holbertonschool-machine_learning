#!/usr/bin/env python3
"""Isolation random tree for outlier detection."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """A random tree whose leaf values encode isolation depth."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize the isolation tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return the textual representation of the whole tree."""
        return self.root.__str__()

    def depth(self):
        """Return the depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree (optionally only the leaves)."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Compute lower/upper bounds for every node of the tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Build a vectorized predict function over the whole tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def np_extrema(self, arr):
        """Return the min and max of a 1D array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random feature and a random threshold for a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf whose value is its depth (isolation depth)."""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node child for a sub-population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively grow a node, stopping only on depth or isolation."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        max_criterion = feature_values > node.threshold
        left_population = node.sub_population & max_criterion
        right_population = node.sub_population & ~max_criterion

        # Is left node a leaf ?
        is_left_leaf = (node.depth + 1 == self.max_depth or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth + 1 == self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train the isolation tree on an explanatory array."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")

#!/usr/bin/env python3
"""Decision tree building blocks with pretty-printing support."""
import numpy as np


class Node:
    """An internal node of a decision tree (feature/threshold split)."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize an internal node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """Prefix the textual representation of the left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Prefix the textual representation of the right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """Return the recursive textual representation of the subtree."""
        if self.is_root:
            text = (f"root [feature={self.feature}, "
                    f"threshold={self.threshold}]\n")
        else:
            text = (f"-> node [feature={self.feature}, "
                    f"threshold={self.threshold}]\n")
        if self.left_child:
            text += self.left_child_add_prefix(
                self.left_child.__str__().rstrip())
        if self.right_child:
            text += self.right_child_add_prefix(
                self.right_child.__str__().rstrip())
        return text

    def max_depth_below(self):
        """Return the maximum depth among the leaves below this node."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node (optionally only the leaves)."""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below() +
                self.right_child.count_nodes_below())

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively attach lower/upper feature bounds to each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child is self.left_child:
                child.lower[self.feature] = max(
                    self.threshold,
                    child.lower.get(self.feature, -np.inf))
            else:
                child.upper[self.feature] = min(
                    self.threshold,
                    child.upper.get(self.feature, np.inf))

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Build the boolean indicator function for this node's region."""
        def is_large_enough(x):
            """True for individuals above every lower bound."""
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower.keys()]),
                axis=0)

        def is_small_enough(x):
            """True for individuals at or below every upper bound."""
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper.keys()]),
                axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict the target for a single individual x (recursive)."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """A leaf node holding a prediction value."""

    def __init__(self, value, depth=None):
        """Initialize a leaf with its value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return the textual representation of the leaf."""
        return (f"-> leaf [value={self.value}]")

    def max_depth_below(self):
        """A leaf's depth is the maximum depth of its branch."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """A leaf counts as a single node/leaf."""
        return 1

    def get_leaves_below(self):
        """Return this leaf in a singleton list."""
        return [self]

    def update_bounds_below(self):
        """A leaf inherits its bounds from its parent; nothing to do."""
        pass

    def pred(self, x):
        """Return this leaf's value for a single individual x."""
        return self.value


class Decision_Tree():
    """A binary decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the tree (optionally around a pre-built root)."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """Return the textual representation of the whole tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute lower/upper bounds for every node of the tree."""
        self.root.update_bounds_below()

    def pred(self, x):
        """Predict the target for a single individual x (recursive)."""
        return self.root.pred(x)

    def update_predict(self):
        """Build a vectorized predict function over the whole tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def depth(self):
        """Return the depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree (optionally only the leaves)."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

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

    def fit(self, explanatory, target, verbose=0):
        """Train the tree on explanatory features and targets."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                                 self.target)}""")

    def fit_node(self, node):
        """Recursively grow a node into children or leaves."""
        node.feature, node.threshold = self.split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        max_criterion = feature_values > node.threshold
        left_population = node.sub_population & max_criterion
        right_population = node.sub_population & ~max_criterion

        # Is left node a leaf ?
        is_left_leaf = (np.sum(left_population) < self.min_pop or
                        node.depth + 1 == self.max_depth or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (np.sum(right_population) < self.min_pop or
                         node.depth + 1 == self.max_depth or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child holding the most represented class."""
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node child for a sub-population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return the fraction of correct predictions on a dataset."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

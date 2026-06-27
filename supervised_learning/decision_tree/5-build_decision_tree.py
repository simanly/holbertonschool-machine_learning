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

    def get_leaves_below(self):
        """Return this leaf in a singleton list."""
        return [self]

    def update_bounds_below(self):
        """A leaf inherits its bounds from its parent; nothing to do."""
        pass


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

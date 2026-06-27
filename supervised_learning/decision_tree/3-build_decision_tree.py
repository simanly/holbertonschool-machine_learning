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

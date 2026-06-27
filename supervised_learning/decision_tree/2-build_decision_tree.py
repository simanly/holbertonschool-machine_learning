#!/usr/bin/env python3
'''
Module that contains classes to build and manage a Decision Tree
'''
import numpy as np


class Node:
    '''
    Represents an internal node in a decision tree
    '''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        '''
        Initializes a new tree internal node
        '''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth of the nodes below this internal node.
        """
        left_max = self.left_child.max_depth_below()
        right_max = self.right_child.max_depth_below()
        return max(left_max, right_max)

    def count_nodes_below(self, only_leaves=False):
        '''
        Recursively counts the nodes below this internal node.
        '''
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        current_count = 0 if only_leaves else 1
        return current_count + left_count + right_count

    def left_child_add_prefix(self, text):
        '''
        Adds structural indentation strings for left child rendering block.
        '''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        '''
        Adds structural indentation strings for right child rendering block.
        '''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        '''
        Returns the formatted string representation of an internal
        decision node
        '''
        if self.is_root:
            result = (f"root [feature={self.feature}, "
                      f"threshold={self.threshold}]\n")
        else:
            result = (f"-> node [feature={self.feature}, "
                      f"threshold={self.threshold}]\n")
        if self.left_child:
            result += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            result += self.right_child_add_prefix(self.right_child.__str__())
        return result


class Leaf(Node):
    '''
    Represents a leaf node in a decision tree.
    '''
    def __init__(self, value, depth=None):
        '''
        Initializes a new leaf node
        '''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        '''
        Returns the depth of the leaf node
        '''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''
        Returns 1 because a leaf node is always counted.
        '''
        return 1

    def __str__(self):
        '''
        Returns the string representation of a leaf node
        '''
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    '''
    Represents a full Decision Tree classifier/regressor
    '''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''
        Initializes the decision tree with hyperparameters
        '''
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

    def depth(self):
        '''
        Returns the overall depth of the decision tree
        '''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''
        Returns the total number of nodes/leaves in the tree
        '''
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        '''
        Returns the tree structure representation string
        '''
        return self.root.__str__()

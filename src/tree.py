from __future__ import annotations
import random
from collections import deque
from copy import deepcopy
from src.utils import (
    concat,
    stride_factor,
    resnet_block1,
    resnet_block2,
    resnet_block3,
    resnet_block4,
    two_times_filters,
    three_times_filters,
)


class Individual(object):
    def __init__(self, tree=None):
        """
        Class that represents a simple Individual having only a GPTree and a fitness value

        :param tree: GPTree object that represents the intrinsics of an Individual
        """
        self.tree = tree
        self.fitness = None


class GPTree(object):
    STRIDES_LIMIT = 5

    # Save all terminals and non-terminals
    nt = [concat, stride_factor, two_times_filters, three_times_filters]
    t = [resnet_block1, resnet_block2, resnet_block3, resnet_block4]
    binary = {concat}

    def __init__(self):
        """
        Class that implements a binary tree. It has some class variables whose save the non-terminals and terminals
        nodes and also register which of the non-terminal are binary operations.

        In this Tree are 3 typse of operations:
            - Terminals: No operation is a constant value (in this case a ResBlock)
            - Unary: Operation that involves only 1 non-terminal/terminal nodes
            - Binary: Operation that involves 2 non-terminal/terminal nodes

        The rule is left is always first! So in unary operations the parent node has only 1 child that is placed always
        in the left side.

        The generation is based in FIFO (First In First Out) so a stack always feeding the left side.

        The 2 rules are fullfilled:
            - If the selected operation a "str" then an terminal is added in its left side
            - If "str" reach the limit (in the article is specified 5 as maximum) then it is removed from the object
              list of non-terminals (nt). That's why a copy of the class list nt is made for each object, to allow
              objects to be prohibited to generate more "str" nodes but others that didn't exceed the limit can
              generate "str" nodes.

        """
        self.parent = None
        self.left = None
        self.right = None
        self.op = None
        self.strides_factor_count = 0

        # Because stride_factor could be deleted but needs to stay in nt list
        # So a copy of nt is created in the object
        self.nt = [i for i in GPTree.nt]

    def __str__(self, spaces=0):
        """
        Pretty print of the tree. Nodes above the left most concat are the left nodes and the others below are the
        right ones. It's generated recursively, traversing the tree inorder.

        :param spaces:
        :return:
        """
        r = ""
        if self.left:
            r += self.left.__str__(spaces + 10)

        r += " " * spaces + self.op.__name__ + '\n'

        if self.right:
            r += self.right.__str__(spaces + 10)

        return r

    def __create_node(self, node: GPTree, branch: str, choices: list):
        """
        Private method that creates a node given a side/branch and list of choices for its value (operation).

        Beacuse the attribute (branch) can be left or right, it's used setattr and getattr.

        :param node: GPTree object to be expanded (creating children)
        :param branch: Branch to expand
        :param choices: List of possible values
        """
        # Set new branch node
        setattr(node, branch, GPTree())

        # Get the new node
        branched = getattr(node, branch)

        # Set its parent as the parameter node
        setattr(branched, 'parent', node)

        # Set the new node with a random op
        setattr(branched, 'op', random.choice(choices))

    def gen_tree(self, max_depth, is_grow=False, init_symbol=None):
        """
        Method that generate a random tree with a max_depth.

        As explianed in generate_feature() method, this can initialize the root node with a specific
        value (init_symbol) and can use Grow method instead of the default one, Full.

        Like was said above (class description) the rules are fulfilled during the tree generation.

        Normally is used the deque data structure for its efficiency push and poping data.

        :param max_depth: Maximum depth of the tree
        :param is_grow: Apply or not the Grow method
        :param init_symbol: Value of the root node, if None is selected randomly from terminals list
        """
        self.op = (
            random.choice(self.nt) if init_symbol is None else init_symbol
        )

        stack = deque([self])
        depth = 0

        # Save all nodes to process on stack
        while stack:

            node = stack.popleft()

            if depth < max_depth:

                if node.op in GPTree.nt:

                    # This condition guarantee that only STRIDES_LIMIT strides_factors are used
                    if (
                        node.op == stride_factor
                        and self.strides_factor_count < GPTree.STRIDES_LIMIT
                    ):
                        self.strides_factor_count += 1

                        # Create a left node that will be a terminal
                        self.__create_node(node, "left", GPTree.t)

                        depth += 1

                        # Remove it from non-terminal, limit is STRIDES_LIMIT
                        if (
                            self.strides_factor_count == GPTree.STRIDES_LIMIT
                            and stride_factor in self.nt
                        ):
                            self.nt.remove(stride_factor)

                    else:
                        # If grow, program select from primitives and terminals
                        if is_grow:
                            nodes = self.nt + GPTree.t
                        else:
                            nodes = self.nt

                        # For binary operations
                        if node.op in GPTree.binary:
                            self.__create_node(node, "right", nodes)
                            stack.appendleft(node.right)

                        # For unary operations
                        self.__create_node(node, "left", nodes)
                        stack.appendleft(node.left)

                        depth += 1

            # Finalizes with a Terminal node
            else:
                node.op = random.choice(GPTree.t)

    def get_nodes(self):
        """
        Method that traverse all tree iteratively and save the nodes in sequence.

        :return: List of GPTree nodes
        """
        stack, nodes = deque([self]), [self]

        while stack:
            node = stack.popleft()

            if node.right:
                stack.appendleft(node.right)
                nodes.append(node.right)
            if node.left:
                stack.appendleft(node.left)
                nodes.append(node.left)

        return nodes

    def count_nodes(self):
        """
        Method that gets the nodes and count them
        :return: Number of nodes in the tree
        """
        return len(self.get_nodes())

    def get_depth_node(self):
        """
        Method that counts the depth counting from the root until the object (self)

        :return: Depth of the current object
        """
        depth = 0
        stack = deque([self])

        while stack:
            node = stack.pop()

            if node.parent is not None:
                depth += 1
                stack.append(node.parent)
            else:
                break

        return depth

    def set_to_node(self, node: GPTree):
        """
        Method that given a GPTree node copy it into the current object (self)

        Attention is needed that children of node were copied into self, so now the children
        have a new parent (in this case is self their new parent), so it's needed to update
        the parents of the children.

        :param node:
        """
        self.op = node.op
        self.right = node.right
        self.left = node.left

        # Set new parents for the swapped children
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self

    def get_num_strides_factor(self):
        """
        Method that counts the number of "str" nodes in the tree
        :return:
        """
        return sum(1 for n in self.get_nodes() if n.op == stride_factor)

    def swap_tree(self, node2):
        """
        Method that given a GPTree node swap it with the current node (self)
        :param node2:
        """
        # parent attribute could be used but the program don't know what node called it
        # left or right ? So the easiest way is to set the values using temporary variable
        tmp = deepcopy(self)
        self.set_to_node(node2)
        node2.set_to_node(tmp)

    def eval_tree(self):
        """
        Method that recursively traverse the object (bottom-up) executing each operation (nodes values).
        :return: List of Resblocks
        """
        if self.op in GPTree.nt:
            if self.op in GPTree.binary:
                return self.op(self.left.eval_tree(), self.right.eval_tree())
            return self.op(self.left.eval_tree())
        return self.op()


class Generic:
    @staticmethod
    def tournment_selection(pop: list[Individual], k: int, tour_size: int):
        """
        Static method that implements a tournament selection. This type of selection gets tour_size
        individuals uniformly from population. Then it will select the K better individuals, in this case
        , based on the fitness value

        :param pop: List of individuals
        :param k: Number of individuals to be selected
        :param tour_size: Number of individuals to be sampled for tournament
        :return: List with winner individuals
        """
        return [max(random.sample(pop, tour_size), key=lambda n: n.fitness) for _ in range(k)]

    @staticmethod
    def roulette_wheel(values: list):
        """
        Static method that implements a roulette wheel where each value has a probability based on how higher
        the value is. Then one value is sampled from that weighted distribution.

        :param values: List of values to be sampled
        :return: Value randomly selected
        """
        norm_factor = sum(values)
        probs = list(map(lambda x: x / norm_factor, values))

        # choices() return a list with 1 element
        return random.choices(range(len(values)), weights=probs)[0]
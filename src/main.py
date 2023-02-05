from __future__ import annotations
from copy import deepcopy
from src.pipeline import Pipeline
from src.model import Tree2Model, train_loop
from src.tree import GPTree, Individual, Generic
from src.utils import concat, stride_factor
import random
import numpy as np


class Search:
    def __init__(
            self,
            gen: int,
            n_pop: int,
            max_depth: int,
            epochs: int,
            mate_prob=0.8,
            mut_prob=0.2,
    ):
        """
        This class have the logic of the genetic algorithm where genetic operators
        are applied and individuals are evaluated

        :param gen: Number of generations
        :param n_pop: Number of individuals
        :param max_depth: Maximum depth of individual's tree
        :param mate_prob: Crossover probability
        :param mut_prob: Mutation probability
        """
        self.gen = gen
        self.n_pop = n_pop
        self.max_depth = max_depth
        self.mate_prob = mate_prob
        self.mut_prob = mut_prob
        self.elitism = 1
        self.epochs = epochs
        self.pop = self.init_population()
        self.best_individual = None

    def common_eval(self, individual: Individual) -> float:
        """
        Evaluation function that receives an Individual, train it and get
        their test accuracy using test dataset.

        To convert and train the individual into a Keras Model the fit_model function
        are used.

        :param individual: Object that represents an individual
        :return: Test accuracy of the individual
        """
        model = Tree2Model(individual.tree.eval_tree())
        _, acc = train_loop(model, self.epochs)

        return acc

    def save_fitness_values(self, fitness_values: list[float], pop: list[Individual],
                            best_individual: Individual) -> Individual:
        """
        Function that save fitness values for each individual and return the
        best individual based on the highest fitness value.

        The lists' fitness_values and pop are in the same order to be possible
        to iterate simultaneously

        :param fitness_values: List of fitness values
        :param pop: List with all individuals
        :param best_individual: Individual object which is the best individual in the previous generation
        :return: New best individual
        """
        # Save fitness values on each individual
        for ind, fit in zip(pop, fitness_values):
            ind.fitness = fit

            # If not initialised yet
            if best_individual is None:
                best_individual = ind
            # Found a better individual
            elif fit > best_individual.fitness:
                best_individual = ind

        return best_individual

    def statistics(self, fitness_values: list[float]):
        """
        Function that uses the class StatWriter to accumulate and print evolution statistics about
        fitnesse values.

        :param fitness_values: List with all fitness values
        """
        min_ind = np.min(fitness_values)
        max_ind = np.max(fitness_values)
        mean = np.mean(fitness_values)
        std = np.std(fitness_values)

        print("Min: {:.3f} Max: {:.3f} Mean: {:.3f} Std: {:.3f}".format(min_ind, max_ind, mean, std))

    def eval_save(self, population: list[Individual], best_individual: Individual) -> Individual:
        """
        Function that call evaluation, statistics and save fitness values functions.

        Returns the best individual which were returned from save_fitness_values function.

        :param population: List with all individuals
        :param best_individual: The best individual in the previous generation
        :return:
        """
        # Evaluate population's individuals
        fitness_values = list(map(self.common_eval, population))

        # Accumulate statistics based on fitness values
        self.statistics(fitness_values)

        # Updates individuals fitness_values and return best individual
        return self.save_fitness_values(
            fitness_values, population, best_individual
        )

    def init_population(self) -> list[Individual]:
        """
        Like showed on GP-CNAS article (https://arxiv.org/ftp/arxiv/papers/1812/1812.07611.pdf), the population
        is created using Ramped-half-and-half initialization (Koza 1994). Here the individuals are randomly generated
        using full and grow methods alternately.

        Full method select a non-terminal nodes until it reached the maximum depth - 1, then put a terminal
        Grow method select a non-terminal or terminal nodes

        We split the population in parts with incremental depth {2,3,4 ... max_depth} until reach the maximum depth.

        Each part applies alternately the 2 methods. This approach allows more diverse trees.

        :return: List of individuals
        """
        pop = []

        parts_size = self.n_pop // (self.max_depth - 1)

        if parts_size <= 0:
            parts_size = 1

        curr_depth = 2
        grow = False

        for i in range(0, self.n_pop, parts_size):
            for j in range(parts_size):
                pop.append(
                    Individual(
                        self.generate_feature(
                            curr_depth, True if grow else False
                        )
                    )
                )
                grow = not grow
            curr_depth += 1

        return pop

    def generate_feature(self, max_depth=None, grow=False, init_symbol=concat) -> GPTree:
        """
        This function is just a wrapper to create a root node (knowing that a node is a Tree itself) and generate
        the rest of the tree based on a max_depth and an initial symbol (value of the root node).

        In the article is specified that the root node is ALWAYS a concatenation primitive. So, by default,
        init_symbol is the function concat.

        If no max_depth are provided it uses the one passed to the Search constructor.

        By default, it uses Full method, using boolean grow it's possible to change to Grow method.

        :param max_depth: Maximum depth for the tree
        :param grow: Boolean to specify the use of Grow method
        :param init_symbol: Root node value
        :return: Root node of the generated tree
        """
        tree = GPTree()
        tree.gen_tree(
            self.max_depth if max_depth is None else max_depth,
            grow,
            init_symbol,
        )
        return tree

    def mutation(self, individual: Individual):
        """
        Mutation was implemented as explained in the article. A random node is selected and a tree
        wit maximum depth of 4 is generated and replace the previous subtree.

        A rule defined on the article says that after a "str" node (double stride node) should be ALWAYS
        a terminal node. Although, here any procedure is needed to prevent that, because the generate_feature
        function which calls gen_tree internally, already prevents that behavior during tree generation.

        Grow method needs to be used.

        :param individual: Mutated individual
        """

        children = individual.tree.get_nodes()

        rand_node = random.choice(children)

        # Generate a new tree with 4 as maximum depth
        max_depth = 4
        new_tree = self.generate_feature(max_depth, True, rand_node.op)

        # Assign the new root node tree to rand_node
        rand_node.set_to_node(new_tree)

    def dynamic_crossover(self, p1: Individual, p2: Individual, gen: int, cur_gen: int):
        """
        Function that implements the proposed dynamic crossover.

        Firstly the root node of both individual are not used during the process, again, because root nodes
        should always be a concatenation primitive.

        Nodes to be swapped are grouped and scored as showed on the article.

        Two kinds of validations was applied (they aren't presented on the article's pseudocode but was assumed
        they're needed):
          - Ensure that "str" nodes always have a terminal node as child
          - After the crossover both trees don't exceed the maximum number of double stride nodes

        In case of one individual has a non-terminal node as root node to swap and the other has a "str" node
        as parent, then the "str" is removed (replacing it by the node to be swapped)

        In case of the individuals exceed the number of double stride nodes the crossover doesn't happen (this
        was the best way I found to prevent this behavior).

        In the end both individuals are swapped in-place.

        :param p1: One individual
        :param p2: Another individual
        :param gen: Total number of generations
        :param cur_gen: Current generation
        :return:
        """

        p1_nodes, p2_nodes = p1.tree.get_nodes(), p2.tree.get_nodes()
        tree1, tree2, p1_nodes, p2_nodes = (
            p1_nodes[0],
            p2_nodes[0],
            p1_nodes[1:],
            p2_nodes[1:],
        )

        nodes_vec = list()
        diff_vec = list()

        # Calculate differences of subtree sizes
        for p1_node in p1_nodes:
            for p2_node in p2_nodes:
                nodes_vec.append([p1_node, p2_node])
                diff = p1_node.count_nodes() - p2_node.count_nodes()
                diff_vec.append(abs(diff))

        # Normalize differences
        min_diff, max_diff = min(diff_vec), max(diff_vec)

        # Only normalize if are different values
        if min_diff != max_diff:
            diff_vec = map(
                lambda x: (x - min_diff) / (max_diff - min_diff), diff_vec
            )

        # Calculate score using differences
        score_vec = [1 + diff * (gen - 2 * cur_gen) / gen for diff in diff_vec]

        # Select a score index to be the crossover point
        cross_pt_index = Generic.roulette_wheel(score_vec)

        # Map to the nodes to swap
        p1_node, p2_node = nodes_vec[cross_pt_index]

        # TODO Put this validation code in separated functions
        # Validate if parent is not a str in case of current node isn't a terminal
        # In this case, copy the op to the str node, clear the children (only left)
        # and re-reference the swap node to their parent
        if not (p1_node.op in GPTree.t) and p2_node.parent.op == stride_factor:
            p2_node.parent.left = None
            p2_node.parent.op = p2_node.op
            p2_node = p2_node.parent

        if not (p2_node.op in GPTree.t) and p1_node.parent.op == stride_factor:
            p1_node.parent.left = None
            p1_node.parent.op = p1_node.op
            p1_node = p1_node.parent

        update1 = (
                tree1.get_num_strides_factor()
                + p2_node.get_num_strides_factor()
                - p1_node.get_num_strides_factor()
        )
        update2 = (
                tree2.get_num_strides_factor()
                + p1_node.get_num_strides_factor()
                - p2_node.get_num_strides_factor()
        )
        if update1 > GPTree.STRIDES_LIMIT or update2 > GPTree.STRIDES_LIMIT:
            print("Crossover will not happen")
            return

        # Update stride count
        tree1.strides_factor_count = update1
        tree2.strides_factor_count = update2

        p1_node.swap_tree(p2_node)

    def search_loop(self):
        """
        Main function where the search is done. Initially all individuals are evaluated and find the best individual.
        Then genetic operators are applied and a new offspring (population) is generated.

        Population size is updated based on elitism size, but the code **only works for elitism=1**

        The tournament size for the tournament selection is calculated based on the formula showed on the GP-CNAS
        article (https://arxiv.org/ftp/arxiv/papers/1812/1812.07611.pdf). When generations are equal to 1 the logarithm
        is 0 and dividing that gives some trouble. To prevent that tour_size was calculated manually for generations = 1
        which give the result 2.

        Deepcopy was used to not change parents in-place, creating a true copy of them and apply crossover in-place.

        Mutation is applied in-place, no deepcopy required.

        """
        best_individual = self.eval_save(self.pop, None)

        # Evolution loop
        for g in range(1, self.gen + 1):
            print("\n-- Generation %d --\n" % g)

            # Calculate the offspring size keeping space for best individuals (elites)
            n_pop = self.n_pop - self.elitism

            # Apply formula and prevent errors with float division with or by zero
            if g == 1 or self.gen == 1:
                tour_size = 2
            else:
                tour_size = np.ceil(
                    2 + (n_pop / 2 - 2) * np.log2(g) / np.log2(self.gen)
                ).astype(int)

            # New population list
            offspring = []

            # Apply crossover using always 2 parents
            for _ in range(n_pop):
                p1, p2 = Generic.tournment_selection(self.pop, 2, tour_size)

                # Make crossover using copies of parents to not change the original ones
                p1_copy, p2_copy = deepcopy(p1), deepcopy(p2)
                self.dynamic_crossover(p1_copy, p2_copy, self.gen, g)

                # Save the new individual (only one child is generated)
                offspring.append(p1_copy)

            # Select nodes to mutate
            for individual in offspring:
                if random.random() < self.mut_prob:
                    self.mutation(individual)

            # Save best individual (deepcopy to prevent in-place changes)
            offspring.append(deepcopy(best_individual))

            # Evaluate and find the new best individual
            self.best_individual = self.eval_save(offspring, best_individual)

            # Older population is replaced by the new offspring
            self.pop[:] = offspring

        print("\n#### Best Individual ####")
        print(
            f"Acc: {self.best_individual.fitness} | Tree: {self.best_individual.tree}"
        )


def main():
    EPOCHS = 5
    GENERATIONS = 1
    MAX_DEPTH = 10
    N_POP = 3

    search = Search(GENERATIONS, N_POP, MAX_DEPTH, EPOCHS)

    search.search_loop()


if __name__ == "__main__":
    main()
from cpt.cpt_functions import CPTFunctions

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from itertools import product
import plotly.graph_objects as go
from scipy import stats

# Class to handle the tree structure of the gamble/trade environments
class GambleTree:

    def __init__(self, game_a_win_val, game_a_lose_val, game_b_win_val, game_b_lose_val, win_probability_a, win_probability_b, generations, alpha_value, delta_value, lambda_value):
        # Initialise all inputs needed for the binary tree set up
        # Win and loss values for each gamble
        self.game_a_win_val = game_a_win_val
        self.game_a_lose_val = game_a_lose_val
        self.game_b_win_val = game_b_win_val
        self.game_b_lose_val = game_b_lose_val

        # Probability of winning each gamble
        self.win_probability_a = win_probability_a
        self.win_probability_b = win_probability_b

        # Number of time steps
        self.generations = generations

        # Alpha parameter value - concavity/convexity parameter for value function
        self.alpha_value = alpha_value

        # Delta parameter value - over/underweighting parameter for probability weighting function
        self.delta_value = delta_value

        # Lambda parameter value - loss aversion parameter
        self.lambda_value = lambda_value

        # Initialise class of CPT functions
        self.CPT = CPTFunctions(alpha_value, lambda_value, delta_value)

    # This function generates the list of outcomes that one could have reached after playing a certain number of times
    def generate_outcomes(self,num_games):
        # Inputs: self.game_a/b_win/lose_val (integers), num_games (integer)
        # Output: cumulative_outcomes (array)

        # Recall possible outcomes for each game and define an array
        game_values = [self.game_a_win_val, self.game_a_lose_val, self.game_b_win_val, self.game_b_lose_val]

        # Generate all sequences of outcomes for num_games
        outcome_sequences = product(game_values, repeat=num_games)

        # Compute the cumulative sums for each sequence
        cumulative_outcomes = [sum(sequence) for sequence in outcome_sequences]

        return cumulative_outcomes

    # This function forms a matrix representing the monetary value a gambler can have at each node
    def base_value_matrix(self):
        # Inputs: self.generations (integer)
        # Output: val_matrix (matrix)

        # Set the size (width) of the matrix and initialise a 1-row zero matrix of this size
        size = 4 ** self.generations
        val_matrix = np.zeros((1, size))

        #Iterate through the time steps
        for i in range(1,self.generations + 1):
            # create an array of possible outcomes for each time step and add 0's to make this array the correct size
            row = self.generate_outcomes(i)
            row_filled = row + [0] * (size - len(row))
            # Add the generated row to the value matrix
            val_matrix = np.vstack([val_matrix, row_filled])

        return val_matrix

    # This function outputs the value matrix after values have been transformed
    def cpt_value_matrix(self, val_matrix):
        # Inputs: val_matrix (matrix), self.generations (integer)
        # Output: transformed_value_matrix (matrix)

        transformed_value_matrix = val_matrix.copy()

        # Iterate through each entry in the matrix and apply the value function
        for i in range(self.generations + 1):
            for j in range(4 ** i):
                #Apply the value function to each matrix entry
                transformed_value_matrix[i,j] = self.CPT.value_function(val_matrix[i,j])

        return transformed_value_matrix

    # This function sets up the matrix of the probabilities of reaching each node
    def initialise_probability_matrix(self):
        # Inputs: generations (integer), win/loss probabilities (integers)
        # Output: probability_matrix (matrix)

        # Initialise a matrix which is 1 in the (0,0) entry and 0 everywhere else
        probability_matrix = np.zeros((self.generations + 1, 4 ** self.generations))  # set all entries to 0
        probability_matrix[0,0] = 1
        multiplier = 1

        # Iterate through each node to update probabilities
        for i in range(1,self.generations+1):
            for j in range(4 ** i):

                # Determine what the probability value is at the parent node
                group = j//4
                prev_val = probability_matrix[i-1,group]

                # Set a multiplier to be applied to the parent probability depending on the node that is reached
                if j % 4 == 0: # Win Game A
                    multiplier = self.win_probability_a
                elif j % 4 == 1: # Lose Game A
                    multiplier = 1-self.win_probability_a
                elif j % 4 == 2: # Win Game B
                    multiplier = self.win_probability_b
                elif j % 4 == 3: # Lose Game B
                    multiplier = 1-self.win_probability_b

                # Apply the multiplier
                probability_matrix[i,j] = prev_val * multiplier

        return probability_matrix

    # This function initialises a blank matrix to be used to store agent strategies
    def initialise_strategy_matrix(self):
        # Input: generations (integer)
        # Output: strategy_matrix (matrix)

        strategy_matrix = np.zeros((self.generations + 1, 4 ** self.generations))
        return strategy_matrix

    # This function returns the possible next-generation outcomes—along with their actions
    @staticmethod
    def gamble_outcomes(current_gen, current_index, strat_matrix):
        # Input: current_gen (integer), current_index (integer), strat_matrix (matrix)
        # Output: outcomes (array)

        # If strategy says stop at current position, return that
        if strat_matrix[current_gen][current_index] == 0:
            return [(current_gen, current_index, "stop")]

        outcomes = []
        next_gen = current_gen + 1

        if next_gen >= len(strat_matrix):
            return outcomes  # No further generation to transition to

        strategy = strat_matrix[current_gen][current_index]

        # Map strategy type to correct win/lose column positions
        if strategy == 1:
            win_col, lose_col = current_index * 4, current_index * 4 + 1
        elif strategy == 2:
            win_col, lose_col = current_index * 4 + 2, current_index * 4 + 3

        # Function to determine the next action
        def get_action(next_strat_value):
            if next_strat_value == 1:
                return "continue_a"
            elif next_strat_value == 2:
                return "continue_b"
            else:
                return "stop"

        # Append outcomes for win and loss branches
        for col in [win_col, lose_col]:
            action = get_action(strat_matrix[next_gen][col])
            outcomes.append((next_gen, col, action))

        return outcomes

    # This function recursively determines all possible nodes that a strategy can reach from a specified node
    def all_outcomes(self,current_gen, current_index, strat_matrix, val_matrix):
        # Inputs: current_gen (integer), current_index (integer), strat_matrix (matrix), val_matrix (matrix)
        # Outputs: outcomes (array), positions (array)

        outcomes = []
        positions = []

        # Determine direct outcomes at starting point
        dir_outcomes = self.gamble_outcomes(current_gen, current_index, strat_matrix)

        # If direct outcomes are "stops" return the outcome value, otherwise we apply a recursion
        for next_row, next_col, status in dir_outcomes:
            if status == "stop":
                outcomes.append(float(val_matrix[next_row][next_col]))
                positions.append((next_row, next_col))
            else:
                sub_values, sub_positions = self.all_outcomes(next_row,next_col, strat_matrix, val_matrix)
                outcomes.extend(sub_values)
                positions.extend(sub_positions)

        # Return the list of outcomes and their node labels
        return outcomes, positions

    # This function determines the probabilities of each outcome
    @staticmethod
    def outcome_probabilities(outcomes, positions, prob_matrix):
        # Inputs: outcomes (array), positions (array), prob_matrix (matrix)
        # Outputs: simplified_outcomes (array), probabilities (array)

        # Remove duplicate outcomes
        seen = set()
        simplified_outcomes = []
        for value in outcomes:
            if value not in seen:
                simplified_outcomes.append(value)
                seen.add(value)

        # Map each unique outcome to its indices in the original list
        index_matches = {x: [i for i, val in enumerate(outcomes) if val == x] for x in simplified_outcomes}

        #Extract individual probabilities from matrix
        raw_probs = [prob_matrix[row, col].item() for row, col in positions]
        total_prob = sum(raw_probs)

        # Aggregate probabilities per outcome
        probabilities = []
        for i in simplified_outcomes:
            prob = sum(raw_probs[j] for j in index_matches[i]) / total_prob
            probabilities.append(prob)

        return simplified_outcomes, probabilities

    # This function performs the above functions sequentially to output a list of outcomes and pi-values
    def pt_lottery_finder(self,t, node, strat, values, probs):
        # Inputs: time (integer), node (integer), strat (matrix), values (matrix), probs (matrix)
        # Outputs: sorted_out (array), pi_vals (array)

        outcomes, positions = self.all_outcomes(t, node, strat, values)
        simp_out, probabilities = self.outcome_probabilities(outcomes, positions, probs)
        sorted_out, pi_vals = self.CPT.pi_values_function(simp_out, probabilities)

        return sorted_out, pi_vals

    # This function takes in a strategy matrix and simplifies it into a plot-able format
    @staticmethod
    def strat_simplify(strategy):
        # Input: strategy (matrix)
        # Output: reduced_matrix (matrix)

        n, n4 = strategy.shape  # Dimensions of the input matrix
        reduced_matrix = np.zeros_like(strategy)  # Start with a zero matrix of the same shape
        reduced_matrix[0] = strategy[0]  # Copy the first row as-is

        for i in range(n - 1):  # Process all rows except the last
            for j in range(n4):
                # Select relevant strategy matrix entries depending on actions
                if strategy[i, j] == 1:
                    reduced_matrix[i + 1, 2 * j] = strategy[i + 1, 4 * j]
                    reduced_matrix[i + 1, 2 * j + 1] = strategy[i + 1, 4 * j + 1]
                elif strategy[i, j] == 2:
                    reduced_matrix[i + 1, 2 * j ] = strategy[i + 1, 4 * j + 2]
                    reduced_matrix[i + 1, 2 * j + 1] = strategy[i + 1, 4 * j + 3]

        # Reduce matrix size
        reduced_matrix = reduced_matrix[0:n,0:(2**(n-1))]

        return reduced_matrix

    # This function plots an individual strategy as a tree
    def plot_tree(self,strategy_matrix):
        G = nx.DiGraph()

        # Dictionary to hold node colors
        node_colours = {}
        labels = {}

        for i in range(self.generations+1):
            for j in range(2**i):
                # Define the node label as (i,j)
                node_label = f"({i},{j})"
                G.add_node(node_label, subset=i)
                labels[node_label] = node_label

                # Determine the color based on the matrix: 0 -> black, 1 -> lightblue
                if strategy_matrix[i][j] == 0:
                    node_colours[node_label] = 'black'
                elif strategy_matrix[i][j] == 1:
                    node_colours[node_label] = 'lightblue'
                elif strategy_matrix[i][j] == 2:
                    node_colours[node_label] = 'red'

                # Add edges for the two children of node (i, j) if not at the last level
                if i < self.generations:
                    G.add_edge(node_label, f"({i + 1},{j*2})")  # "down" move
                    G.add_edge(node_label, f"({i + 1},{(2*j) + 1})")  # "up" move

        # Draw the binomial tree with specified colors
        pos = nx.multipartite_layout(G, subset_key="subset", align="vertical")  # Layout algorithm for positioning nodes

        nx.draw(G, pos, with_labels=True, node_color=[node_colours[node] for node in G.nodes],
                node_size=350, font_size=7, font_weight='bold', edge_color='gray', arrows=True)

        font_colours = {node: 'white' if node_colours[node] == 'black' else 'black' for node in G.nodes}
        for node, (x, y) in pos.items():
            plt.text(x, y, labels[node], fontsize=7, fontweight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     color=font_colours[node])
        plt.savefig(f'figures/tree_{int(time.time())}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # This function performs the no-commitment sophisticate backwards induction
    def backward_induction(self, plot = False):
        # Inputs: self
        # Output: clean_strat (matrix), plot

        # Initialise the matrices we use
        strategy = self.initialise_strategy_matrix()
        value_matrix = self.base_value_matrix()
        value_func_matrix = self.cpt_value_matrix(value_matrix)
        prob_matrix = self.initialise_probability_matrix()
        generations = self.generations

        # iterate through nodes from the end of the tree to the root node
        for i in reversed(range(generations)):
            for j in range(4**i):
                # Set the action at the node to 1 and evaluate
                strategy[i,j] = 1
                sorted_a_outcomes, a_pi_values = self.pt_lottery_finder(i,j,strategy, value_func_matrix, prob_matrix) #weight probabilities
                a_pt_gamble_value = np.sum(np.multiply(sorted_a_outcomes, a_pi_values)) #calculate prospect theory value of gamble

                # Set the action at the node to 1 and evaluate
                strategy[i, j] = 2
                sorted_b_outcomes, b_pi_values = self.pt_lottery_finder(i,j,strategy, value_func_matrix, prob_matrix)  # weight probabilities
                b_pt_gamble_value = np.sum(np.multiply(sorted_b_outcomes, b_pi_values))  # calculate prospect theory value of gamble

                # Select action that results in highest value
                if a_pt_gamble_value <= value_func_matrix[i][j] and b_pt_gamble_value <= value_func_matrix[i][j]:
                    strategy[i,j] = 0 #stop if pt vlaue is lower than value fo the decision node
                elif value_func_matrix[i][j] < a_pt_gamble_value and b_pt_gamble_value <= a_pt_gamble_value:
                    strategy[i,j] = 1 #stop if pt vlaue is greater than value fo the decision node
                elif value_func_matrix[i][j] < b_pt_gamble_value and a_pt_gamble_value < b_pt_gamble_value:
                    strategy[i,j] = 2

        clean_strat = self.strat_simplify(strategy)
        if plot:
            self.plot_tree(clean_strat)

        return clean_strat

    # This function generates all possible feasible strategies from a specified node
    def generate_strategies_quad_tree_multi_choice(self, gens, start_index):
        # Inputs: gens (integer), start_index (integer)
        # Output: valid_strategies (array)

        num_decision_nodes = (2 ** gens) - 1  # Total decision nodes in the tree
        valid_strategies = []

        # This sub-function determines if a strategy is feasible
        def is_valid_combination(decisions):
            index = 0
            # Track parent presence for each generation
            prev_gen = [1]  # Root node must be 1

            for i in range(gens):
                curr_gen = []
                for j in range(2**i):
                    decision = decisions[index]
                    index += 1
                    # A node can only be 1 or 2 if at least one parent exists
                    if decision == 1 or decision == 2:
                            if prev_gen[j // 2] == 0:
                                return False
                    curr_gen.append(decision)

                # Move to the next generation
                prev_gen = curr_gen
            return True

        # Generate and check combinations of strategies
        for decision_combination in product([0, 1, 2], repeat=num_decision_nodes):
            if is_valid_combination(decision_combination):
                # Create the corresponding matrix
                matrix = np.zeros((gens + 1, 4**gens), dtype=int)
                index = 0
                for i in range(gens):
                    for j in range(2**i):
                        matrix[i, j] = decision_combination[index]
                        index += 1

                larger_matrix = np.zeros((gens + 1, 4 ** gens), dtype=int)
                larger_matrix[0] = matrix[0]

                for i in range(gens):  # Process all rows except the last
                    for j in range(4**i):
                        # Expand generated list of actions onto a matrix of correct size
                        if matrix[i, j] == 1:  # Action 1
                            larger_matrix[i + 1, 4 * j] = matrix[i + 1, 2 * j]
                            larger_matrix[i + 1, 4 * j + 1] = matrix[i + 1, 2 * j + 1]
                        elif matrix[i, j] == 2:  # Action 2
                            larger_matrix[i + 1, 4 * j + 2] = matrix[i + 1, 2 * j]
                            larger_matrix[i + 1, 4 * j + 3] = matrix[i + 1, 2 * j + 1]
                valid_strategies.append(larger_matrix)

        # If we are generating a sub tree, place it in the correct location of a regular tree
        if gens < self.generations:
            adjusted_strategies = []
            for matrix in valid_strategies:
                valid_matrix = np.zeros((self.generations + 1, self.generations ** 4), dtype=int)
                valid_matrix[self.generations - gens:, start_index:start_index + 4**gens] = matrix
                adjusted_strategies.append(valid_matrix)
            return adjusted_strategies

        return valid_strategies

    # This function yields the CPT maximising strategy from a given node
    def optimal_strategy_find(self,start_gen, start_index):
        # Inputs: start_gen (integer), start_index (integer)
        # Outputs: optimal_strategy (matrix), max_val (float)

        # Generate all possible strategies and initialise matricies
        strategies = self.generate_strategies_quad_tree_multi_choice(self.generations-start_gen, start_index)
        values = []
        value_matrix = self.base_value_matrix()
        value_func_matrix = self.cpt_value_matrix(value_matrix)
        prob_matrix = self.initialise_probability_matrix()

        # For each strategy, calculate the pt value and add this value to a list
        for idx in range(len(strategies)):
            strategy = strategies[idx]
            sorted_outcomes, pi_values = self.pt_lottery_finder(start_gen, start_index, strategy, value_func_matrix,prob_matrix)
            pt_gamble_value = np.sum(np.multiply(sorted_outcomes, pi_values))
            values.append(pt_gamble_value)

        # Determine the maximum CPT value achieved by any strategy and the corresponding strategy matrix
        max_val = max(values)
        optimal_strategy = [mat for mat, val in zip(strategies, values) if val == max_val] #return strategy with highest value

        return optimal_strategy[0], max_val

    # This function performs the algorithm to determine the commited sophisticates strategy
    def commitment_sophisticated(self, plot=False):
        # Inputs: self
        # Output: optimal_strategy (matrix), plot

        # Decision is made at (0,0)
        start_gen = 0
        start_index = 0

        #Compute optimal strategy and format the matrix for plotting
        optimal_strategy, val = self.optimal_strategy_find(start_gen, start_index) #determine optimal strategy
        clean_strat = self.strat_simplify(optimal_strategy)
        if plot:
            self.plot_tree(clean_strat)

        return clean_strat

    # This strategy perfomrs the algorithm to determine the naive agent strategy
    def naive(self, plot=False):
        # Inputs: self
        # Output: true_strategy (matrix), plot

        true_strategy = np.zeros((self.generations + 1, 4 ** self.generations))

        # Determines what strategy the naive agent enters with
        planned_strategy = self.commitment_sophisticated(plot=False)
        true_strategy[0][0] = planned_strategy[0][0]

        # Based on past actions iterate through reachable nodes and re-evaluate strategy
        for i in range(1, self.generations + 1):
            for j in range(4**i):
                # If certain actions are taken, some nodes become unreachable
                if (j % 4 == 0 or j % 4  == 1) and true_strategy[i - 1][j//4] != 1:
                    true_strategy[i][j] = 0
                elif (j % 4 == 2 or j % 4  == 3) and true_strategy[i - 1][j//4] != 2:
                    true_strategy[i][j] = 0
                # Evaluate new best strategy at each node
                else:
                    optimal_strategy, val = self.optimal_strategy_find(i,j)
                    if optimal_strategy[i][j] == 1:
                        true_strategy[i][j] = 1
                    elif optimal_strategy[i][j] == 2:
                        true_strategy[i][j] = 2

        # Format matrix for plotting
        clean_strat = self.strat_simplify(true_strategy)
        if plot:
            self.plot_tree(clean_strat)

        return clean_strat

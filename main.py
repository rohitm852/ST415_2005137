from cpt.multi_game_tree_class import GambleTree
from cpt.simulations_class import Simulations
from cpt.cpt_functions import CPTFunctions
import numpy as np

def main():
    # Define parameters for the CPT and game setup
    game_a_win_val = 10
    game_a_lose_val = -10
    game_b_win_val = 20
    game_b_lose_val = -10

    win_probability_a = 0.4865
    win_probability_b = 0.3243

    generations = 3

    alpha_value = 0.95    # CPT value function exponent
    lambda_value = 1.5    # CPT loss aversion
    delta_value = 0.5   # CPT probability weighting

    # Initialize the GambleTree with parameters
    tree = GambleTree(
        game_a_win_val,
        game_a_lose_val,
        game_b_win_val,
        game_b_lose_val,
        win_probability_a,
        win_probability_b,
        generations,
        alpha_value,
        delta_value,
        lambda_value
    )
    sims = Simulations(
        game_a_win_val,
        game_a_lose_val,
        game_b_win_val,
        game_b_lose_val,
        win_probability_a,
        win_probability_b,
        generations)


    #strategy_matrix_cs = tree.commitment_sophisticated(plot=True)
    #strategy_matrix_n = tree.naive(plot=True)
    #strategy_matrix_ncs = tree.backward_induction(plot=True)

    #print("Strategy matrix:")
    #print(strategy_matrix)

    #entry_data = sims.parameter_test_cs(20)
    #entry_data = sims.parameter_test_no_cs(20)

    np.random.seed(2005137)
    cs_a, cs_b, cs_c = sims.simulations_cs(10000)
    np.random.seed(2005137)
    ncs_a, ncs_b, ncs_c = sims.simulations_bi(10000)
    np.random.seed(2005137)
    n_a, n_b, n_c = sims.simulations_naive(10000)

if __name__ == "__main__":
    main()
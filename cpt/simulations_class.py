from cpt.multi_game_tree_class import GambleTree

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import plotly.graph_objects as go
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, to_hex, to_rgb



# Class to run simulations based on the tree model to analyse entries and behaviour
class Simulations:

    def __init__(self, game_a_win_val, game_a_lose_val, game_b_win_val, game_b_lose_val, win_probability_a, win_probability_b, generations):
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

    # This function produces a 3D Interactive plot of casino entries
    @staticmethod
    def plot_param_chart(df):
        plot_df = df
        color_map = {1: 'blue', 2: 'red'}
        plot_df["Colour"] = plot_df["Result"].map(color_map)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=plot_df["Alpha"],
            y=plot_df["Delta"],
            z=plot_df["Lambda"],
            mode='markers',
            marker=dict(size=1, color=plot_df["Colour"], symbol='x'),
        ))
        camera = dict(
            eye=dict(x=0.5, y=1.2, z=0.1)
        )
        # Set Labels
        fig.update_layout(
            title="Interactive 3D Scatter Plot",
            scene_camera=camera,
            scene=dict(
                xaxis=dict(title="Alpha", autorange="reversed"),
                yaxis=dict(title="Delta", autorange="reversed"),
                zaxis=dict(title="Lambda", ),
                aspectmode='cube'
            )
        )

        fig.show()

    # This function produces a parameter grid
    def _build_param_grid(self, n):
        alphas = np.linspace(0, 1, n)
        deltas = np.linspace(0.3, 1, n)
        lambdas = np.linspace(1, 4, n)
        return list(product(alphas, deltas, lambdas))

    # This function initialises game parameters
    def _get_game_args(self):
        return (
            self.game_a_win_val,
            self.game_a_lose_val,
            self.game_b_win_val,
            self.game_b_lose_val,
            self.win_probability_a,
            self.win_probability_b
        )

    # This function outputs the entry behaviour given parameters of the no commitment agent
    @staticmethod
    def entry_nc(a, d, l, game_args, generations):
        tree = GambleTree(*game_args, generations, a, d, l)
        strat = tree.backward_induction(plot=False)
        return (a, d, l, strat[0][0])

    # This function outputs the entry behaviour given parameters of the naive and commitment agent
    @staticmethod
    def entry_commitment(a, d, l, game_args, generations):
        tree = GambleTree(*game_args, generations, a, d, l)
        strat = tree.commitment_sophisticated(plot=False)
        return (a, d, l, strat[0][0])

    # This function models entry behaviour of the no commitment agent
    def parameter_test_no_cs(self, n, plot=True, n_jobs=-1):
        # Initialise game parameters and build parameter grid
        param_grid = self._build_param_grid(n)
        game_args = self._get_game_args()

        # Compute entries in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.entry_nc)(a, d, l, game_args, self.generations)
            for (a, d, l) in tqdm(param_grid, total=len(param_grid), desc="Running grid search")
        )

        df = pd.DataFrame(results, columns=["Alpha", "Delta", "Lambda", "Result"])
        entry_df = df[df["Result"] != 0]

        if plot:
            self.plot_param_chart(entry_df)

        return entry_df



    # This function models entry behaviour of the naive and commitment agent
    def parameter_test_cs(self, n, plot=True, n_jobs=-1):
        # Initialise game parameters and build parameter grid
        param_grid = self._build_param_grid(n)
        game_args = self._get_game_args()

        # Compute entries in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.entry_commitment)(a, d, l, game_args, self.generations)
            for (a, d, l) in tqdm(param_grid, total=len(param_grid), desc="Running grid search")
        )

        df = pd.DataFrame(results, columns=["Alpha", "Delta", "Lambda", "Result"])
        entry_df = df[df["Result"] != 0]
        if plot:
            self.plot_param_chart(entry_df)

        return entry_df

    # This function samples from a truncated normal distribution - bounded
    @staticmethod
    def truncated_normal(mean, std, lower, upper, size=1000):
        # Inputs: mean, std, lower and upper bounds
        # Output: samples
        a, b = (lower - mean) / std, (upper - mean) / std
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        return samples

    # This function randomly samples parameters
    def sample_params(self,n):
        # Input: n
        # Outputs: alphas, deltas, lambdas
        alphas = self.truncated_normal(0.88, 0.25, 0, 1, n)
        deltas = self.truncated_normal(0.65, 0.125, 0, 1, n)
        lambdas = np.random.normal(2.5, 0.5, n)
        lambdas = np.where(lambdas > 1, lambdas, 1 + np.abs(lambdas - 1))

        return alphas, deltas, lambdas

    # This function plots aggregated simulation results on a tree
    @staticmethod
    def plot_tree_sim(results_a, results_b, results_s, n):
        G = nx.DiGraph()
        node_colours = {}
        labels = {}
        idx = 0

        diffs = results_b - results_a
        max_diff = max(abs(diffs.min()), abs(diffs.max()), 1e-5)
        reds = cm.Reds
        blues = cm.Blues

        for i in range(4):
            for j in range(2 ** i):
                if i < 3:
                    node_label = f"{idx}:A:{results_a[idx]:.0f}\nB:{results_b[idx]:.0f}"
                    display_label = f"A:{results_a[idx]:.0f}\nB:{results_b[idx]:.0f}"
                    G.add_node(node_label, subset=i)
                    labels[node_label] = display_label

                    if results_s[idx] == n:
                        node_colours[node_label] = 'black'
                    else:
                        diff = results_b[idx] - results_a[idx]
                        if diff > 0:
                            normalized = diff / max_diff
                            node_colours[node_label] = to_hex(reds(0.3 + 0.7 * normalized))
                        elif diff < 0:
                            normalized = -diff / max_diff
                            node_colours[node_label] = to_hex(blues(0.3 + 0.7 * normalized))
                        else:
                            node_colours[node_label] = '#cccccc'
                else:
                    node_label = f"{j}"
                    G.add_node(node_label, subset=i)
                    node_colours[node_label] = 'black'
                    labels[node_label] = node_label

                left_child_idx = 2 * idx + 1
                right_child_idx = 2 * idx + 2

                if i < 2:
                    left_label = f"{left_child_idx}:A:{results_a[left_child_idx]:.0f}\nB:{results_b[left_child_idx]:.0f}"
                    right_label = f"{right_child_idx}:A:{results_a[right_child_idx]:.0f}\nB:{results_b[right_child_idx]:.0f}"
                    G.add_edge(node_label, left_label)
                    G.add_edge(node_label, right_label)

                if i == 2:
                    G.add_edge(node_label, f"{2 * j}")
                    G.add_edge(node_label, f"{2 * j + 1}")

                idx += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.multipartite_layout(G, subset_key="subset")

        def brightness(hex_color):
            r, g, b = to_rgb(hex_color)
            return 0.299 * r + 0.587 * g + 0.114 * b

        edge_colours = []
        for node in G.nodes:
            bg_color = node_colours[node]
            if node.isdigit():
                edge_colours.append('black')  # leaf node
            else:
                edge_colours.append('black' if brightness(bg_color) > 0.9 else bg_color)

        nx.draw(
            G, pos, ax=ax,
            with_labels=False,
            node_color=[node_colours[node] for node in G.nodes],
            edgecolors=edge_colours,
            linewidths=1.5,
            node_size=1300,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrows=True
        )

        for node, (x, y) in pos.items():
            bg_color = node_colours[node]
            node_brightness = brightness(bg_color)
            text_color = 'white' if node_brightness <= 0.4 or bg_color == 'black' else 'black'

            ax.text(x, y, labels[node], fontsize=10, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center',
                    color=text_color)

        plt.tight_layout()
        plt.show()

    # This function plots histograms of the distributions of the sampled parameters
    @staticmethod
    def params_hists(alphas, deltas, lambdas):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(alphas, bins=50, density=True, alpha=0.6, color='b')
        plt.title('Distribution Alpha')
        plt.xlabel('Value')
        plt.ylabel('Density')

        plt.subplot(1, 3, 2)
        plt.hist(deltas, bins=50, density=True, alpha=0.6, color='g')
        plt.title('Distribution of Delta')
        plt.xlabel('Value')

        plt.subplot(1, 3, 3)
        plt.hist(lambdas, bins=50, density=True, alpha=0.6, color='r')
        plt.title('Distribution of Lambda')
        plt.xlabel('Value')

        plt.tight_layout()
        plt.show()

    # This function returns both tree plot and histograms
    def simulation_plots(self, results_a, results_b, results_s, n, alphas, deltas, lambdas):
        self.plot_tree_sim(results_a, results_b, results_s, n)
        self.params_hists(alphas, deltas, lambdas)

    # This function generates a backwards induction strategy
    @staticmethod
    def run_simulation_bi(a, d, l, game_args,generations):
        tree = GambleTree(*game_args, generations, a, d, l)
        return tree.backward_induction(plot=False)

    # This function aggregates all backward induction strategies for the simulated parameters
    def simulations_bi(self, n, n_jobs=-1, plot = True):
        # Initialise parameters
        alphas, deltas, lambdas = self.sample_params(n)
        game_args = self._get_game_args()
        strategies = []

        # Obtain strategies in parallel
        all_strats = Parallel(n_jobs=n_jobs)(
            delayed(self.run_simulation_bi)(alphas[i], deltas[i], lambdas[i], game_args, self.generations)
            for i in tqdm(range(n), desc="Running NCS Simulations")
        )

        for i in range(n):
            strat = all_strats[i]
            if strat[0][0] !=0:
                strategies.append(strat)
        # Store number of each action taken in each node
        results_a = np.zeros(7)
        results_b = np.zeros(7)
        results_s = np.zeros(7)
        idx = 0
        for t in range(3):
            for j in range(2 ** t):
                entries = [matrix[t, j] for matrix in strategies]
                for k in range(len(strategies)):
                    result = entries[k]

                    if result == 0:
                        results_s[idx] += 1
                    elif result == 1:
                        results_a[idx] += 1
                    elif result == 2:
                        results_b[idx] += 1

                idx += 1
        if plot:
            self.simulation_plots(results_a, results_b, results_s, len(strategies), alphas, deltas, lambdas)

        return results_a, results_b, results_s

    # This function generates a commited sophisticate strategy
    @staticmethod
    def run_simulation_cs(a, d, l, game_args, generations):
        tree = GambleTree(*game_args, generations, a, d, l)
        return tree.commitment_sophisticated(plot=False)

    # This function aggregates all commited sophisticate strategies for the simulated parameters
    def simulations_cs(self, n, n_jobs=-1, plot = True):
        # Initialise parameters
        alphas, deltas, lambdas = self.sample_params(n)
        game_args = self._get_game_args()

        # Obtain strategies in parallel
        strategies = Parallel(n_jobs=n_jobs)(
            delayed(self.run_simulation_cs)(alphas[i], deltas[i], lambdas[i], game_args, self.generations)
            for i in tqdm(range(n), desc="Running CS Simulations")
        )

        # Store number of each action taken in each node
        results_a = np.zeros(7)
        results_b = np.zeros(7)
        results_s = np.zeros(7)
        idx = 0
        for t in range(3):
            for j in range(2 ** t):
                entries = [matrix[t, j] for matrix in strategies]
                for k in range(n):
                    result = entries[k]

                    if result == 0:
                        results_s[idx] += 1
                    elif result == 1:
                        results_a[idx] += 1
                    elif result == 2:
                        results_b[idx] += 1

                idx += 1
        if plot:
            self.simulation_plots(results_a, results_b, results_s, n, alphas, deltas, lambdas)

        return results_a, results_b, results_s

    # This function generates a naive strategy
    @staticmethod
    def run_simulation_naive(a, d, l, game_args, generations):
        tree = GambleTree(*game_args, generations, a, d, l)
        return tree.naive(plot=False)

    # This function aggregates all naive strategies for sampled parameters
    def simulations_naive(self, n, n_jobs=-1, plot=True):
        # Initialise the parameters
        alphas, deltas, lambdas = self.sample_params(n)
        game_args = self._get_game_args()

        # Determine the naive strategies in parallel
        strategies = Parallel(n_jobs=n_jobs)(
            delayed(self.run_simulation_naive)(alphas[i], deltas[i], lambdas[i], game_args, self.generations)
            for i in tqdm(range(n), desc="Running N Simulations")
        )

        # Store number of each action taken at each node
        results_a = np.zeros(7)
        results_b = np.zeros(7)
        results_s = np.zeros(7)
        idx = 0
        for t in range(3):
            for j in range(2 ** t):
                entries = [matrix[t, j] for matrix in strategies]
                for k in range(n):
                    result = entries[k]

                    if result == 0:
                        results_s[idx] += 1
                    elif result == 1:
                        results_a[idx] += 1
                    elif result == 2:
                        results_b[idx] += 1

                idx += 1
        if plot:
            self.simulation_plots(results_a, results_b, results_s, n, alphas, deltas, lambdas)

        return results_a, results_b, results_s





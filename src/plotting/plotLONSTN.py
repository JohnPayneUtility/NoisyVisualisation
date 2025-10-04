# --------------------
# IMPORTS
# --------------------

import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px  # for continuous color scales
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import TSNE

from .DashboardHelpers import *

from ..problems.FitnessFunctions import *
from ..problems.ProblemScripts import load_problem_KP

# --------------------
# SETTINGS
# --------------------


# --------------------
# CODE
# --------------------

def update_plot(optimum, PID, opt_goal, options, run_options, STN_lower_fit_limit,
                LO_fit_percent, LON_options, LON_node_colour_mode, LON_edge_colour_feas,
                NLON_fit_func, NLON_intensity, NLON_samples, layout_value, plot_type,
                hover_info_value, azimuth_deg, elevation_deg, all_trajectories_list, STN_labels,
                run_start_index, n_runs_display, local_optima, axis_values,
                opacity_noise_bar, LON_node_opacity, LON_edge_opacity, STN_node_opacity, STN_edge_opacity,
                STN_node_min, STN_node_max, LON_node_min, LON_node_max,
                LON_edge_size_slider, STN_edge_size_slider, noisy_fitnesses_list):
    print('Running plotting function...')
    print(STN_labels)
    # LON Options
    LON_filter_negative = 'LON-filter-neg' in LON_options
    LON_hamming = 'LON-hamming' in LON_options
    # Options from checkboxes
    show_labels = 'show_labels' in options
    hide_STN_nodes = 'hide_STN_nodes' in options
    hide_LON_nodes = 'hide_LON_nodes' in options
    plot_3D = 'plot_3D' in options
    use_solution_iterations = 'use_solution_iterations' in options
    LON_node_strength = 'LON_node_strength' in options
    local_optima_color = 'local_optima_color' in options

    # Run options
    show_best = 'show_best' in run_options
    show_mean = 'show_mean' in run_options
    show_median = 'show_median' in run_options
    show_worst = 'show_worst' in run_options
    STN_hamming = 'STN-hamming' in run_options

    # Options from dropdowns
    layout = layout_value

    # G = nx.DiGraph()
    G = nx.MultiDiGraph()

    # Colors for different sets of trajectories
    algo_colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    # algo_colors = ['blue', 'purple', 'cyan', 'magenta', 'brown']
    # algo_colors = ['magenta', 'brown', 'teal']
    node_color_shared = 'green'
    option_curve_edges = True

    # Add nodes and edges for each set of trajectories
    stn_node_mapping = {}
    lon_node_mapping = {}

    def generate_run_summary_string(selected_trajectories):
        lines = []
        for run_idx, entry in enumerate(selected_trajectories):
            if len(entry) != 5:
                lines.append(f"Skipping malformed entry in run {run_idx}")
                continue
            unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry
            # Convert noisy fitnesses to ints if needed:
            noisy_fitnesses = [int(fit) for fit in noisy_fitnesses]
            lines.append(f"Run {run_idx}:")
            for i, solution in enumerate(unique_solutions):
                lines.append(f"  Solution: {solution} | Fitness: {unique_fitnesses[i]} | Noisy Fitness: {noisy_fitnesses[i]}")
            lines.append("")  # Blank line between runs
        return "\n".join(lines)
    
    def print_hamming_transitions(all_run_trajectories, print_sols=False, print_transitions=False):
        """
        For each run in all_run_trajectories, print the normalized Hamming distance
        between consecutive solutions, then print min, max, and median for that run.
        Finally, print overall min, max, and median across all runs.
        
        all_run_trajectories: list of runs,
        where each run is a tuple/list of 
        (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)
        """
        overall_distances = []
        
        for run_idx, entry in enumerate(all_run_trajectories):
            if len(entry) != 5:
                print(f"Run {run_idx}: Skipping malformed entry (expected 5 elements, got {len(entry)})")
                continue
                
            unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry
            distances = []
            
            print(f"Run {run_idx}:")
            for i in range(len(unique_solutions) - 1):
                sol1 = unique_solutions[i]
                sol2 = unique_solutions[i+1]
                dist = hamming_distance(sol1, sol2)
                distances.append(dist)
                overall_distances.append(dist)
                if print_transitions:
                    print(f"  Transition from solution {i} to {i+1}:")
                if print_sols:
                    print(f"    {sol1} -> {sol2}")
                print(f"    Hamming distance: {dist:.3f}")
            
            if distances:
                run_min = min(distances)
                run_max = max(distances)
                run_median = np.median(distances)
                print(f"  Run {run_idx} summary:")
                print(f"    Min: {run_min:.3f}, Max: {run_max:.3f}, Median: {run_median:.3f}")
            else:
                print("  No transitions found.")
                
            print("")  # Blank line between runs

        if overall_distances:
            overall_min = min(overall_distances)
            overall_max = max(overall_distances)
            overall_median = np.median(overall_distances)
            print("Overall summary across runs:")
            print(f"    Min: {overall_min:.3f}, Max: {overall_max:.3f}, Median: {overall_median:.3f}")
        else:
            print("No transitions found overall.")
    
    def add_trajectories_to_graph(all_run_trajectories, edge_color, algo_idx):
        # print_hamming_transitions(all_run_trajectories)
        for run_idx, entry in enumerate(all_run_trajectories):
            # Check data length and None values as before...
            if len(entry) != 5:
                print(f"Skipping malformed entry {entry}, expected 5 elements but got {len(entry)}")
                continue
            unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry
            noisy_fitnesses = [int(fit) for fit in noisy_fitnesses]
            if any(x is None for x in (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)):
                print(f"Skipping run {run_idx} due to None values: {entry}")
                continue

            # Create nodes and store node labels in order for this run
            # node_labels = []  # to store the node labels in order
            for i, solution in enumerate(unique_solutions):
                current_fitness = unique_fitnesses[i]
                if STN_lower_fit_limit is not None:
                    if current_fitness < STN_lower_fit_limit:
                        # Skip adding this node (and its noisy node and edge) because its fitness is below the threshold.
                        continue
                start_node = True if i == 0 else False
                end_node = True if i == len(unique_solutions) - 1 else False
                solution_tuple = tuple(solution)
                key = (solution_tuple, "STN")
                if key not in stn_node_mapping:
                    node_label = f"STN_{len(stn_node_mapping) + 1}"
                    # node_label = f'Algo{algo_idx}_Run{run_idx}_Step{i}'
                    # STN_node_label = f'STN_{node_label}'
                    stn_node_mapping[key] = node_label
                    G.add_node(node_label, solution=solution, fitness=unique_fitnesses[i], 
                               iterations=solution_iterations[i], type="STN", run_idx=run_idx, step=i,
                               color=edge_color, start_node=start_node, end_node=end_node)
                    # print(f"DEBUG: Added STN node {node_label} for solution {solution_tuple}")
                else:
                    node_label = stn_node_mapping[key]
                    print(f"DEBUG: Reusing STN node {node_label} for solution {solution_tuple}")

                # Add noisy node for STN data (if desired)
                noisy_node_label = f"Noisy_{node_label}"
                if noisy_node_label not in G.nodes():
                    try:
                        G.add_node(noisy_node_label, solution=solution, fitness=noisy_fitnesses[i], color=edge_color)
                        # print(f"DEBUG: Added noisy node {noisy_node_label}")
                    except Exception as e:
                        print(f"Error adding noisy node: {noisy_node_label}, {e}")
                    G.add_edge(node_label, noisy_node_label, weight=STN_edge_size_slider, 
                               color=edge_color, edge_type='Noise')
                    # print(f"DEBUG: Added Noise edge from {node_label} to {noisy_node_label}")
            # Add transitions as STN edges
            for j, (prev_solution, current_solution) in enumerate(transitions):
                prev_key = (tuple(prev_solution), "STN")
                curr_key = (tuple(current_solution), "STN")
                if prev_key in stn_node_mapping and curr_key in stn_node_mapping:
                    src = stn_node_mapping[prev_key]
                    tgt = stn_node_mapping[curr_key]
                    G.add_edge(src, tgt, weight=STN_edge_size_slider, color=edge_color, edge_type='STN')
                    # print(f"DEBUG: Added STN edge from {src} to {tgt}")

    debug_summaries = []
    # Add trajectory nodes if provided
    if all_trajectories_list:
        # Determine optimisation goal
        
        optimisation_goal = opt_goal[:3].lower() # now handled via data, update in rest of code

        # Add all sets of trajectories to the graph
        # print(f"Checking all_trajectories_list: {all_trajectories_list}")
        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            # print(f"Checking all_run_trajectories: {all_run_trajectories}")
            edge_color = algo_colors[idx % len(algo_colors)]  # Cycle through colors if there are more sets than colors

            selected_trajectories = []
            if n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[run_start_index:run_start_index+n_runs_display])
            if show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_trajectories_to_graph(selected_trajectories, edge_color, idx)

            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)
    else:
        debug_summary_component = html.Div("No trajectory data available.")

    print('STN TRAJECTORIES ADDED')
        # # Find the overall best solution across all sets of trajectories
        # if optimisation_goal == "max":
        #     overall_best_fitness = max(
        #         max(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _, _ in all_run_trajectories
        #     )
        # else:  # Minimisation
        #     overall_best_fitness = min(
        #         min(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _, _ in all_run_trajectories
        #     )
    
    # CoLON colour helpers
    def sol_tuple_ints(sol):
        # convert any iterable into a tuple of ints
        return tuple(int(x) for x in sol)

    def sol_key_str(sol):
        # "1,0,1,..." string form used in the Store
        return ",".join(str(int(x)) for x in sol)

    def lookup_map(mapp, sol):
        t = sol_tuple_ints(sol)
        s = sol_key_str(sol)
        if isinstance(mapp, dict):
            if t in mapp:
                return mapp[t]
            if s in mapp:
                return mapp[s]
        return None
    
    # Add LON Nodes
    node_noise = {}
    if local_optima:
        # Code for colouring of CoLON components
        opt_feas_map = {}
        neigh_feas_map = {}
        if local_optima and isinstance(local_optima, dict):
            opt_feas_map = local_optima.get("opt_feas_map", {}) or {}
            neigh_feas_map = local_optima.get("neigh_feas_map", {}) or {}

        node_colour_mode = LON_node_colour_mode  # 'fitness' | 'feasible' | 'neigh'
        colour_edges_by_feas = ('edge_feas' in LON_edge_colour_feas) if isinstance(LON_edge_colour_feas, list) else False
        # -> continued older code for core functionality
        local_optima = convert_to_single_edges_format(local_optima)
        # local_optima = pd.DataFrame(local_optima).apply(convert_to_single_edges_format, axis=1)
        local_optima = filter_local_optima(local_optima, LO_fit_percent)
        if LON_filter_negative:
            local_optima = filter_negative_LO(local_optima)
        # print("DEBUG: Number of local optima:", len(local_optima["local_optima"]))
        
        # ------
        # add nodes for LON
        for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
            solution_tuple = tuple(opt)
            key = (solution_tuple, "LON")
            if key not in lon_node_mapping:
                node_label = f"Local Optimum {len(lon_node_mapping) + 1}"
                lon_node_mapping[key] = node_label
                G.add_node(node_label, solution=opt, fitness=fitness, type="LON")
                # print(f"DEBUG: Added LON node {node_label} for solution {solution_tuple}")
            else:
                node_label = lon_node_mapping[key]

            # NOISE CLOUD FOR LON
            # for i in range(10):
            #     from FitnessFunctions import eval_noisy_kp_v1
            #     from ProblemScripts import load_problem_KP
            #     n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP('f1_l-d_kp_10_269')
            #     noisy_node_label = f"Noisy {node_label} {i+1}"
            #     noisy_fitness = eval_noisy_kp_v1(opt, items_dict=items_dict, capacity=capacity, noise_intensity=1)[0]

            #     noisy_node_size = 15 
            #     G.add_node(noisy_node_label, solution=opt, fitness=noisy_fitness, color='pink', size=noisy_node_size)
            #     # Add an edge from the LON node to this noisy node
            #     # G.add_edge(node_label, noisy_node_label, weight=STN_edge_size_slider, color='pink', style='dotted')

            # NOISE BOX PLOTS FOR LON
            # NLON_fit_func, NLON_intensity,
            node_noise[node_label] = []  # create an empty list for this node's noisy fitness values
            n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(PID)
            for i in range(NLON_samples):
                # Compute the noisy fitness
                if NLON_fit_func == 'kpv1s':
                    noisy_fitness = eval_noisy_kp_v1_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=NLON_intensity)[0]
                elif NLON_fit_func == 'kpv2s':
                    noisy_fitness = eval_noisy_kp_v2_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=NLON_intensity)[0]
                elif NLON_fit_func == 'kpv1mw':
                    noisy_fitness = eval_noisy_kp_v1(opt, items_dict=items_dict, capacity=capacity, noise_intensity=NLON_intensity)[0]
                elif NLON_fit_func == 'kpv2mw':
                    noisy_fitness = eval_noisy_kp_v2(opt, items_dict=items_dict, capacity=capacity, noise_intensity=NLON_intensity)[0]
                elif NLON_fit_func == 'kpp':
                    noisy_fitness, _ = eval_noisy_kp_prior(opt, items_dict=items_dict, capacity=capacity, noise_intensity=NLON_intensity)[0]
                else:
                    print('NO NOISY FITNESS FUNCTION PROVIDED')
                    noisy_fitness = 0
                node_noise[node_label].append(noisy_fitness)
        fitness_dict = {node: data['fitness'] for node, data in G.nodes(data=True)} # for noise box plots
        # print("DEBUG: node_noise keys:", list(node_noise.keys()))

        # Add LON edges
        for (source, target), weight in local_optima["edges"].items():
            source_tuple = tuple(source)
            target_tuple = tuple(target)
            src_key = (source_tuple, "LON")
            tgt_key = (target_tuple, "LON")
            if src_key in lon_node_mapping and tgt_key in lon_node_mapping:
                src_label = lon_node_mapping[src_key]
                tgt_label = lon_node_mapping[tgt_key]
                # G.add_edge(src_label, tgt_label, weight=weight, color='black', edge_type='LON') # previous line before coloured CoLONs
                edge_color = 'black'  # default
                if colour_edges_by_feas and opt_feas_map:
                    tgt_sol = G.nodes[tgt_label].get('solution', [])
                    feas = lookup_map(opt_feas_map, tgt_sol)
                    if feas is not None:
                        edge_color = 'green' if int(feas) == 1 else 'red'

                G.add_edge(src_label, tgt_label, weight=weight, color=edge_color, edge_type='LON')
                # print(f"DEBUG: Added LON edge from {src_label} to {tgt_label}")
        
        # ONLY recolor by weight if we're NOT colouring by feasibility
        if not colour_edges_by_feas:
            # Calculate min and max edge weight for lON for normalisation
            LON_edge_weight_all = [data.get('weight', 2)
                for u, v, key, data in G.edges(data=True, keys=True)
                if "Local Optimum" in u and "Local Optimum" in v]
            if LON_edge_weight_all:
                LON_edge_weight_min = min(LON_edge_weight_all)
                LON_edge_weight_max = max(LON_edge_weight_all)
            else:
                LON_edge_weight_min = LON_edge_weight_max = 1 # set to 1 if LON_edge_weight_all is empty

            # Normalise edge weights for edges between Local Optimum nodes and colour
            for u, v, key, data in G.edges(data=True, keys=True):
                if "Local Optimum" in u and "Local Optimum" in v:
                    weight = data.get('weight', 2) # get un-normalised weight
                    # Normalize the weight (if all weights are equal, default to 0.5)
                    norm_weight = (weight - LON_edge_weight_min) / (LON_edge_weight_max - LON_edge_weight_min) if LON_edge_weight_max > LON_edge_weight_min else 0.5
                    norm_weight = np.clip(norm_weight, 0, 0.9999) # clip normalised weight
                    color = px.colors.sample_colorscale('plasma', norm_weight)[0]
                    data['norm_weight'] = norm_weight
                    data['color'] = color
    
    print('LOCAL OPTIMA ADDED')

    # Normalise solution iterations
    stn_iterations = [
    G.nodes[node].get('iterations', 1)
        for node in G.nodes()
        if "STN" in node
    ]
    if stn_iterations:
        min_STN_iter = min(stn_iterations)
        max_STN_iter = max(stn_iterations)

    # Assign node sizes
    for node, data in G.nodes(data=True):
        if "Local Optimum" in node:
            # For LON nodes: weight is the sum of incoming edge weights.
            incoming_edges = G.in_edges(node, data=True)
            node_weight = sum(edge_data.get('weight', 0) for _, _, edge_data in incoming_edges)
            node_size = LON_node_min + node_weight * (LON_node_max - LON_node_min)
            G.nodes[node]['weight'] = node_weight
            # if data.get('fitness') == optimum:
                # node_size = LON_node_max
        elif "STN" in node:
            # For STN nodes: weight comes from the 'iterations' attribute.
            # node_weight = G.nodes[node].get('iterations', 1)
            iter = G.nodes[node].get('iterations', 1)
            # Normalize to the 0-1 range
            norm_iter = (iter - min_STN_iter) / (max_STN_iter - min_STN_iter) if max_STN_iter > min_STN_iter else 0.5
            node_size = STN_node_min + norm_iter * (STN_node_max - STN_node_min)
            if data.get('fitness') == optimum and "Noisy" not in node:
                node_size = STN_node_max
        else:
            # For any other node, assign a default weight of 1.
            node_size = 1
        # Set the computed weight as a node property.
        G.nodes[node]['size'] = node_size
    
    # Compute fitness range among LON nodes for 'fitness' mode colouring
    local_optimum_nodes = [node for node in G.nodes() if "Local Optimum" in node]
    if local_optimum_nodes:
        all_fitness = [G.nodes[node]['fitness'] for node in local_optimum_nodes]
        min_fit = min(all_fitness)
        max_fit = max(all_fitness)
    else:
        min_fit = max_fit = 0.0

    # node colours
    for node, data in G.nodes(data=True):
        if "STN" in node:
            if data.get('start_node', False):
                data['color'] = 'yellow'
            elif data.get('end_node', False):
                data['color'] = 'brown'
            continue

        if "Local Optimum" in node:
            sol_tuple = tuple(int(x) for x in data.get('solution', []))

            if node_colour_mode == 'fitness':
                # continuous colourscale across LON fitness range
                if max_fit > min_fit:
                    norm = (float(data['fitness']) - min_fit) / (max_fit - min_fit)
                else:
                    norm = 0.5
                data['color'] = px.colors.sample_colorscale('plasma', float(np.clip(norm, 0.0, 0.9999)))[0]

            elif node_colour_mode == 'feasible':
                feas = lookup_map(opt_feas_map, sol_tuple)
                data['color'] = ('green' if int(feas) == 1 else 'red') if feas is not None else 'grey'

            elif node_colour_mode == 'neigh':
                p = lookup_map(neigh_feas_map, sol_tuple)
                data['color'] = px.colors.sample_colorscale('RdYlGn', float(np.clip(p, 0.0, 0.9999)))[0] if p is not None else 'grey'

            else:
                data['color'] = 'grey'

            # keep the "optimum = red" override ONLY when in fitness mode
            if node_colour_mode == 'fitness' and data.get('fitness') == optimum and "Noisy" not in node:
                data['color'] = 'red'
    
    print('LOCAL OPTIMA STATS CALCULATED')
    # LON STATS
    local_optimum_nodes = [node for node in G.nodes() if "Local Optimum" in node]
    if local_optimum_nodes:
        mean_in_degree_local = sum(G.in_degree(node) for node in local_optimum_nodes) / len(local_optimum_nodes)
        mean_out_degree_local = sum(G.out_degree(node) for node in local_optimum_nodes) / len(local_optimum_nodes)
        num_local_optima = len(local_optimum_nodes)
        mean_weight = (
            sum(G.nodes[node].get('weight', 0) for node in local_optimum_nodes) / num_local_optima
            if num_local_optima > 0 else 0
        )
        max_weight = max(G.nodes[node].get('weight', 0) for node in local_optimum_nodes) if num_local_optima > 0 else 0
        mean_fitness = (
        sum(G.nodes[node].get('fitness', 0) for node in local_optimum_nodes) / num_local_optima
        if num_local_optima > 0 else 0
        )
        max_fitness = max(G.nodes[node].get('fitness', 0) for node in local_optimum_nodes) if num_local_optima > 0 else 0
        # Edge stats
        local_optimum_edges = [
            (u, v) for u, v, data in G.edges(data=True)
            if "Local Optimum" in u and "Local Optimum" in v
        ]
        local_optimum_edge_weights = [
            data.get('weight', 0) 
            for u, v, data in G.edges(data=True)
            if "Local Optimum" in u and "Local Optimum" in v
        ]
        if local_optimum_edge_weights:
            max_edge_weight = max(local_optimum_edge_weights)
            mean_edge_weight = sum(local_optimum_edge_weights) / len(local_optimum_edge_weights)
        else:
            max_edge_weight = 0
            mean_edge_weight = 0

        print('num_local_optima', num_local_optima)
        print('max_fitness', max_fitness)
        print(f'mean_fitness: {mean_fitness:.2f}')
        print(f'mean_weight: {mean_weight:.2f}')
        print(f'max_weight: {max_weight:.2f}')
        print(f'mean_in_degree_local: {mean_in_degree_local:.2f}')
        print(f'mean_out_degree_local: {mean_out_degree_local:.2f}')
        print("num_edges:", len(local_optimum_edges))
        print(f'max_edge_weight: {max_edge_weight:.2f}')
        print(f'mean_edge_weight: {mean_edge_weight:.2f}')




    print('\033[32mNode Sizes and Colours Assigned\033[0m')
    print('\033[33mCalculating node positions...\033[0m')
    def normed_hamming_distance(sol1, sol2):
        L = len(sol1)
        return sum(el1 != el2 for el1, el2 in zip(sol1, sol2)) / L
    
    def canonical_solution(sol):
        # Force every element to be an int (adjust as needed)
        return tuple(int(x) for x in sol)

    # Prepare node positions based on selected layout
    # unique_solution_positions = {}
    # solutions = []
    # for node, data in G.nodes(data=True):
    #     sol = tuple(data['solution'])
    #     if sol not in unique_solution_positions:
    #         solutions.append(sol)
    print('\033[33mCompiling Solutions...\033[0m')
    solutions_set = set()
    for node, data in G.nodes(data=True):
        # 'solution' is your bit-string (tuple) stored in the node attributes
        # sol = tuple(data['solution'])
        sol = canonical_solution(data['solution'])
        solutions_set.add(sol)
    solutions_list = list(solutions_set)
    n = len(solutions_list)
    # print("DEBUG: Number of unique solutions for Positioning:", n)
    if n == 0:
        # print("ERROR: No solutions for Positioning")
        pos = {}
    elif layout == 'mds':
        print('\033[33mUsing MDS\033[0m')
        dissimilarity_matrix = np.zeros((len(solutions_list), len(solutions_list)))
        for i in range(len(solutions_list)):
            for j in range(len(solutions_list)):
                dissimilarity_matrix[i, j] = hamming_distance(solutions_list[i], solutions_list[j])

        mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42)
        positions_2d = mds.fit_transform(dissimilarity_matrix)

        solution_positions = {}
        for i, sol in enumerate(solutions_list):
            solution_positions[sol] = positions_2d[i]
        
        pos = {}
        for node, data in G.nodes(data=True):
            sol = tuple(data['solution'])
            # All nodes with the same bit-string get the same (x,y)
            pos[node] = solution_positions[sol]
    elif layout == 'tsne':
        print('\033[33mUsing TSNE\033[0m')
        # Use t-SNE to position nodes based on dissimilarity (Hamming distance)
        # solutions = [data['solution'] for _, data in G.nodes(data=True)]
        # n = len(solutions)
        dissimilarity_matrix = np.zeros((len(solutions_list), len(solutions_list)))
        for i in range(len(solutions_list)):
            for j in range(len(solutions_list)):
                dissimilarity_matrix[i, j] = hamming_distance(solutions_list[i], solutions_list[j])

        # Initialize and fit t-SNE
        tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
        positions_2d = tsne.fit_transform(dissimilarity_matrix)

        solution_positions = {}
        for i, sol in enumerate(solutions_list):
            solution_positions[sol] = positions_2d[i]
        
        pos = {}
        for node, data in G.nodes(data=True):
            sol = tuple(data['solution'])
            # All nodes with the same bit-string get the same (x,y)
            pos[node] = solution_positions[sol]
    # elif layout == 'raw':
        # Directly use the 2D solution values as positions
        # solutions = [data['solution'] for _, data in G.nodes(data=True)]
        # pos = {node: solutions[i] for i, node in enumerate(G.nodes())}
    elif layout == 'kamada_kawai':
        print('\033[33mUsing Kamada Kawai\033[0m')
        # pos = nx.kamada_kawai_layout(G, dim=2)
        # # Update positions for noisy nodes
        # for node in G.nodes():
        #     if node.startswith("Noisy_"):
        #         # Extract the corresponding solution node name by removing the "Noisy " prefix.
        #         solution_node = node.replace("Noisy_", "", 1)
        #         if solution_node in pos:
        #             pos[node] = pos[solution_node]
        
        # 1. Calculate initial force-directed positions on the full graph G
        initial_pos = {}
        try:
            # You might need to adjust parameters like max_iter if the graph is large/complex
            initial_pos = nx.kamada_kawai_layout(G, dim=2, scale=1)
            print(f"Kamada-Kawai initial layout calculated for {len(initial_pos)} nodes.")
        except Exception as e:
            print(f"Kamada-Kawai layout on full graph G failed: {e}")
            print("Falling back to random positions.")
            # Fallback: Assign random positions if KK fails
            # Important: Need to ensure 'pos' is assigned even in failure case
            pos = {node: (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1) for node in G.nodes()}
            initial_pos = None # Signal that KK failed and averaging should be skipped

        # Proceed with averaging only if initial_pos was successfully calculated
        if initial_pos:
            # 2. Group nodes by solution and collect their initial positions
            positions_by_solution = {}
            nodes_without_solution_or_pos = []

            for node, data in G.nodes(data=True):
                # Check if node got an initial position (it should have if KK didn't fail)
                node_pos = initial_pos.get(node)
                if node_pos is None:
                    print(f"Warning: Node {node} missing from initial KK position results.")
                    nodes_without_solution_or_pos.append(node)
                    continue # Skip if no initial position

                # Retrieve the solution attribute safely
                sol_data = data.get('solution')

                if sol_data is not None:
                    try:
                        sol_tuple = canonical_solution(sol_data)
                        if sol_tuple not in positions_by_solution:
                            positions_by_solution[sol_tuple] = []
                        positions_by_solution[sol_tuple].append(node_pos) # Store (x, y) tuple
                    except Exception as e_conv:
                        print(f"Warning: Could not process solution for node {node}: {e_conv}.")
                        nodes_without_solution_or_pos.append(node)
                else:
                    # Keep track of nodes that genuinely lack a solution attribute
                    nodes_without_solution_or_pos.append(node)

            print(f"Processed {len(initial_pos)} nodes. Found {len(positions_by_solution)} unique solutions with positions.")
            if nodes_without_solution_or_pos:
                print(f"Found {len(nodes_without_solution_or_pos)} nodes without a 'solution' attribute or missing from initial positions.")

            # 3. Calculate the average position for each unique solution
            final_solution_positions = {}
            for sol_tuple, pos_list in positions_by_solution.items():
                if not pos_list: continue
                avg_pos = np.mean(np.array(pos_list), axis=0)
                final_solution_positions[sol_tuple] = tuple(avg_pos) # Store as tuple

            # 4. Assign the final (averaged) position to all nodes in G
            pos = {} # Initialize final position dictionary
            assigned_count = 0
            unassigned_count = 0
            for node, data in G.nodes(data=True):
                sol_data = data.get('solution')
                assigned = False
                if sol_data is not None:
                    try:
                        sol_tuple = canonical_solution(sol_data)
                        if sol_tuple in final_solution_positions:
                            pos[node] = final_solution_positions[sol_tuple]
                            assigned = True
                            assigned_count += 1
                    except Exception:
                        pass # Error processing solution handled before

                if not assigned:
                    # Assign original KK position if node had no solution or its solution wasn't processed
                    pos[node] = initial_pos.get(node, (np.random.rand()*0.1, np.random.rand()*0.1)) # Use initial pos or small random fallback
                    unassigned_count += 1
                    # if node in nodes_without_solution_or_pos:
                    #      print(f"Node {node} (no solution/pos issue) assigned its initial KK position or fallback.")

            print(f"Assigned final positions to {assigned_count} nodes based on averaged solution positions.")
            if unassigned_count > 0:
                 print(f"Assigned initial/fallback positions to {unassigned_count} nodes (no solution/pos issue).")

        # Ensure 'pos' dictionary exists even if KK failed initially
        elif 'pos' not in locals():
             pos = {node: (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1) for node in G.nodes()}
            
    elif layout == 'kamada_kawai_weighted':
        print('\033[33mUsing Kamada Kawai\033[0m')
        # pos = nx.kamada_kawai_layout(G, dim=2 if not plot_3D else 3)
        # 2. Build a complete graph of unique solutions:
        CG = nx.complete_graph(n)  # nodes will be 0,1,...,n-1
        mapping = {i: solutions_list[i] for i in range(n)}
        # 3. For each pair, set the edge weight to be the normalized Hamming distance:
        for i in range(n):
            for j in range(i+1, n):
                weight = hamming_distance(solutions_list[i], solutions_list[j])
                CG[i][j]['weight'] = weight
                # Since H is undirected, this weight is used for both directions.

        # 4. Compute the Kamada-Kawai layout on H using the weight attribute.
        pos_unique = nx.kamada_kawai_layout(CG, weight='weight', dim=2)
        
        # 5. Map unique solution positions back to a dictionary keyed by the actual solution tuple.
        solution_positions = { mapping[i]: pos_unique[i] for i in range(n) }
        
        # 6. For every node in G, assign the position corresponding to its solution.
        pos = {}
        for node, data in G.nodes(data=True):
            sol = tuple(data['solution'])
            pos[node] = solution_positions[sol]
    else:
        pos = nx.spring_layout(G, dim=2 if not plot_3D else 3)
    # print("DEBUG: Positions computed for nodes:", pos)
        # Update positions for noisy nodes
        for node in G.nodes():
            if node.startswith("Noisy_"):
                # Extract the corresponding solution node name by removing the "Noisy " prefix.
                solution_node = node.replace("Noisy_", "", 1)
                if solution_node in pos:
                    pos[node] = pos[solution_node]

    print('\033[32mNode Positions Calculated\033[0m')

    # create node_hover_text which holds node hover text information
    node_hover_text = []
    if hover_info_value == 'fitness':
        node_hover_text = [str(G.nodes[node]['fitness']) for node in G.nodes()]
    elif hover_info_value == 'iterations':
        node_hover_text = [str(G.nodes[node]['iterations']) for node in G.nodes()]
    elif hover_info_value == 'solutions':
        node_hover_text = [str(G.nodes[node]['solution']) for node in G.nodes()]


# ---------- PLOTTING -----------
    print('CREATING PLOT...')
    # # Debugging
    # print("DEBUG: Total nodes in G:", len(G.nodes()))
    # print("DEBUG: Nodes and their properties:")
    # for node in G.nodes():
    #     print("  Node:", node, "Properties:", G.nodes[node])
        
    # print("DEBUG: Total edges in G:", len(G.edges()))
    # for u, v, key, data in G.edges(data=True, keys=True):
    #     print("  Edge from", u, "to", v, "Key:", key, "Properties:", data)
    
    # stn_edge_count = sum(1 for u, v, key, data in G.edges(data=True, keys=True) if "STN" in data.get("edge_type", ""))
    # lon_edge_count = sum(1 for u, v, key, data in G.edges(data=True, keys=True) if "LON" in data.get("edge_type", ""))
    # print("DEBUG: STN edge count:", stn_edge_count, "LON edge count:", lon_edge_count)

    if plot_type == 'RegLon' or plot_type == 'NLon_box':
        # Compute a dynamic H based on the fitness range of local optimum nodes
        local_optimum_nodes = [node for node in G.nodes() if 'Local Optimum' in node]
        if local_optimum_nodes:
            all_fitness = [G.nodes[node]['fitness'] for node in local_optimum_nodes]
            fitness_range = max(all_fitness) - min(all_fitness)
        else:
            fitness_range = 1
        # For example, let H be 10% of the overall fitness range; adjust as needed.
        H = fitness_range * 1  
        dx = 0.05  # horizontal offset for the mini boxplot

        traces = []
        edge_traces = []
        edge_label_x = []
        edge_label_y = []
        edge_label_z = []
        edge_labels = []
        edge_counts = {}

        ### Pre-calculate color indices for STN edge pairs
        stn_edges_by_pair_color = {}
        for u_pre, v_pre, key_pre, data_pre in G.edges(data=True, keys=True):
            if data_pre.get("edge_type") == "STN":
                pair_pre = (u_pre, v_pre)
                color_pre = data_pre.get('color', 'default_color') # Use the edge's specific color
                if pair_pre not in stn_edges_by_pair_color:
                    stn_edges_by_pair_color[pair_pre] = {}
                if color_pre not in stn_edges_by_pair_color[pair_pre]:
                    stn_edges_by_pair_color[pair_pre][color_pre] = 0 # Just need to know the color exists
        
        # Create a mapping from color to index for each pair for consistent ordering
        color_indices_for_pair = {}
        for pair, colors_dict in stn_edges_by_pair_color.items():
            # Sort colors alphabetically or by appearance order if needed
            sorted_colors = sorted(colors_dict.keys())
            color_indices_for_pair[pair] = {color: idx for idx, color in enumerate(sorted_colors)}

        # LOOP THROUGH ALL EDGES FOR PLOTTING
        print('Plotting edges...')
        processed_stn_keys = set() # Keep track of multi-edges processed by color logic

        for u, v, key, data in G.edges(data=True, keys=True):
            # Skip edges if nodes don't have positions
            if u not in pos or v not in pos:
                 print(f"Skipping edge ({u}, {v}) as node positions are missing.")
                 continue
            # Skip edges involving hidden node types
            node_u_type = G.nodes[u].get('type')
            node_v_type = G.nodes[v].get('type')
            if (hide_LON_nodes and ("LON" in node_u_type or "LON" in node_v_type)) or \
               (hide_STN_nodes and ("STN" in node_u_type or "STN" in node_v_type or "NoisySTN" in node_u_type or "NoisySTN" in node_v_type)):
                continue


            edge_type = data.get("edge_type", "")
            edge_color = data.get('color', 'grey') # Default edge color
            edge_opacity = 1.0 # Default opacity
            mid_x, mid_y, mid_z = 0, 0, 0 # Initialize midpoints

            # Determine opacity based on edge type
            if edge_type == 'LON':
                edge_opacity = LON_edge_opacity
            elif edge_type == 'STN':
                edge_opacity = STN_edge_opacity
            elif edge_type == 'Noise':
                edge_opacity = STN_edge_opacity # Make noise edges fainter

            # Get start and end points (2D position + Z fitness)
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            # Use 0 if fitness is missing? Or skip edge? Let's use 0 for now.
            z0 = G.nodes[u].get('fitness', 0)
            z1 = G.nodes[v].get('fitness', 0)
            # Ensure fitness values are numeric
            z0 = float(z0) if z0 is not None else 0
            z1 = float(z1) if z1 is not None else 0


            option_curve_edges_LON = False
            # Process curved STN edges
            if option_curve_edges and edge_type == "STN" and (u, v) in color_indices_for_pair:
                 pair = (u, v)
                 current_edge_color = data.get('color', 'default_color') # Color of this specific edge instance
                 
                 color_indices_map = color_indices_for_pair[pair]
                 total_distinct_colors = len(color_indices_map)

                 if current_edge_color in color_indices_map:
                     color_idx = color_indices_map[current_edge_color]

                     base_curvature = 0.2  # Base amount of curve
                     max_offset_factor = 1.5 # How much to spread curves (adjust as needed)

                     if total_distinct_colors > 1:
                         # Spread curvatures symmetrically around 0
                         # The range will be roughly [-base_curvature * max_offset_factor, +base_curvature * max_offset_factor]
                         curvature = base_curvature * max_offset_factor * ( (color_idx - (total_distinct_colors - 1) / 2.0) / ((total_distinct_colors - 1) / 2.0) )
                         # Add a minimum curve even for the middle one if total is odd > 1
                         if abs(curvature) < 0.01: curvature = 0.05 * np.sign(color_idx - (total_distinct_colors - 1) / 2.0 + 1e-6) # Small curve
                     elif total_distinct_colors == 1:
                         # Single color for this transition, use base curvature
                         curvature = base_curvature
                     else: # Should not happen if pair is in color_indices_for_pair
                         curvature = 0

                     # Prevent extremely large curvatures if start/end points are very close
                     dist_xy = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                     if dist_xy < 0.1: # If points are very close in XY plane
                         curvature *= (dist_xy / 0.1) # Scale down curvature


                     start_2d = (x0, y0)
                     end_2d = (x1, y1)
                     # Generate curve points (only needs 2D)
                     curve_xy = quadratic_bezier(start_2d, end_2d, curvature=curvature, n_points=20)

                     # Interpolate Z values along the curve
                     z_values = np.linspace(z0, z1, len(curve_xy))

                     edge_trace = go.Scatter3d(
                         x=list(curve_xy[:, 0]),
                         y=list(curve_xy[:, 1]),
                         z=list(z_values),
                         mode='lines',
                         line=dict(width=STN_edge_size_slider, color=current_edge_color), # Use edge's specific color
                         opacity=edge_opacity,
                         hoverinfo='none',
                         showlegend=False
                     )
                     traces.append(edge_trace)

                     # Midpoint for potential label placement on the curve
                     mid_index = len(curve_xy) // 2
                     mid_x = curve_xy[mid_index, 0]
                     mid_y = curve_xy[mid_index, 1]
                     mid_z = z_values[mid_index]

                 else:
                     # Should not happen if pre-calculation is correct, but handle defensively
                     print(f"Warning: Edge color {current_edge_color} not found for pair {pair}. Drawing straight.")
                     # Fallback to straight line
                     edge_trace = go.Scatter3d(
                         x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines',
                         line=dict(width=STN_edge_size_slider, color=current_edge_color, dash='dot'), # Indicate fallback
                         opacity=edge_opacity * 0.8, hoverinfo='none', showlegend=False
                     )
                     traces.append(edge_trace)
                     mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

            elif option_curve_edges_LON and edge_type == "LON":
                start = pos[u][:2]
                end = pos[v][:2]             
                curvature = 0.2

                dash_style = 'solid'
                norm_w = data.get('norm_weight', 0) # From 0 to 1
                line_width = 1 + norm_w * (LON_edge_size_slider - 1) # Scale between 1 and max size
                edge_color = data.get('color', 'black') # Use color calculated from weight

                # Compute the curved path using your quadratic_bezier function.
                curve = quadratic_bezier(start, end, curvature=curvature, n_points=20)
                z0 = G.nodes[u]['fitness']
                z1 = G.nodes[v]['fitness']
                z_values = np.linspace(z0, z1, len(curve))
                edge_trace = go.Scatter3d(
                    x=list(curve[:, 0]),
                    y=list(curve[:, 1]),
                    z=list(z_values),
                    mode='lines',
                    line=dict(width=max(0.5, line_width), # Ensure minimum width
                              color=edge_color,
                              dash=dash_style),
                    opacity=edge_opacity,
                    hoverinfo='none',
                    showlegend=False
                )
                # For curved edges, choose the midpoint from the curve.
                mid_index = len(curve) // 2
                mid_x = curve[mid_index, 0]
                mid_y = curve[mid_index, 1]
                traces.append(edge_trace)
            else:
                # Draw straight lines for LON, Noise edges, or STN if curving is off/failed
                line_width = 1 # Default
                dash_style = 'solid'
                if edge_type == 'LON':
                    # Use normalized weight for LON edge width (scaled)
                    norm_w = data.get('norm_weight', 0) # From 0 to 1
                    line_width = 1 + norm_w * (LON_edge_size_slider - 1) # Scale between 1 and max size
                    edge_color = data.get('color', 'black') # Use color calculated from weight
                elif edge_type == 'Noise':
                    line_width = 3
                    dash_style = 'solid'
                    edge_color = data.get('color', 'grey') # Should inherit STN node color
                elif edge_type == 'STN': # Straight STN edge
                     line_width = STN_edge_size_slider
                     edge_color = data.get('color', 'green') # Should inherit run color


                edge_trace = go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(width=max(0.5, line_width), # Ensure minimum width
                              color=edge_color,
                              dash=dash_style),
                    opacity=edge_opacity,
                    hoverinfo='none',
                    showlegend=False
                )
                traces.append(edge_trace)
                mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

            # Add edge labels (Hamming distance) if requested
            if should_label_edge(u, v, STN_hamming, LON_hamming):
                try:
                    sol_u = G.nodes[u].get('solution')
                    sol_v = G.nodes[v].get('solution')
                    if sol_u and sol_v: # Check if solutions exist
                        hd = hamming_distance(sol_u, sol_v)
                        edge_label_x.append(mid_x)
                        edge_label_y.append(mid_y)
                        edge_label_z.append(mid_z + 0.1) # Slight offset for visibility
                        edge_labels.append(f"H={hd}") # Add prefix for clarity
                except Exception as e:
                     print(f"Error calculating Hamming distance for label ({u}, {v}): {e}")


        # Create and add a single trace for all edge labels
        if edge_labels:
            edge_label_trace = go.Scatter3d(
                x=edge_label_x, y=edge_label_y, z=edge_label_z,
                mode='text', text=edge_labels,
                textposition="middle center",
                textfont=dict(size=10, color='black'),
                hoverinfo='none',
                showlegend=False
            )
            traces.append(edge_label_trace)

        # ----- Add node trace (without labels) -----
        print('Plotting nodes...')
        node_x, node_y, node_z = [], [], []
        node_sizes, node_colors = [], []

        LON_node_x, LON_node_y, LON_node_z = [], [], []
        LON_node_sizes, LON_node_colors = [], []

        for node, attr in G.nodes(data=True):
            # pos[node] might be a tuple of (x, y) or (x, y, z). Use the first two coordinates for x and y.
            x, y = pos[node][:2]
            z = attr['fitness']
                
            if "Local Optimum" in node:
                LON_node_x.append(x)
                LON_node_y.append(y)
                LON_node_z.append(z)
                LON_node_sizes.append(attr.get('size', 1))
                LON_node_colors.append(attr.get('color', 'grey'))
            else:
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_sizes.append(attr.get('size', 1))
                node_colors.append(attr.get('color', 'blue'))

        # print("Node colors for trace:", node_colors)
        LON_node_trace = go.Scatter3d(
            x=LON_node_x,
            y=LON_node_y,
            z=LON_node_z,
            mode='markers',
            marker=dict(
                size=LON_node_sizes,
                color=LON_node_colors,
                opacity=LON_node_opacity  # Use your desired LON node opacity here.
            ),
            showlegend=False
        )
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',  # markers only; no text labels
            marker=dict(
                size=node_sizes,
                color=node_colors
            ),
            opacity=STN_node_opacity,
            showlegend=False
        )
        traces.append(LON_node_trace)
        traces.append(node_trace)

        # ----- Add mini boxplots for each node using the noise data -----
        # (Only add for nodes that have noise data in node_noise.)
        if plot_type == 'NLon_box':
            print('Plotting noise bar plots...')
            for node in pos:
                if node in fitness_dict and node in node_noise:
                    x, y = pos[node][:2]
                    base_z = fitness_dict[node]
                    noise = np.array(node_noise[node])
                    
                    # Compute quartiles and extremes for the noisy fitness values
                    min_val = np.min(noise)
                    q1 = np.percentile(noise, 25)
                    med = np.median(noise)
                    q3 = np.percentile(noise, 75)
                    max_val = np.max(noise)
                    
                    # Map the noise values linearly to a local z range around the node's base fitness.
                    if max_val == min_val:
                        z_min = z_q1 = z_med = z_q3 = z_max = base_z
                    else:
                        # Scaled boxes
                        # z_min = base_z - H/2
                        # z_max = base_z + H/2
                        # z_q1 = base_z - H/2 + (q1 - min_val) / (max_val - min_val) * H
                        # z_med = base_z - H/2 + (med - min_val) / (max_val - min_val) * H
                        # z_q3 = base_z - H/2 + (q3 - min_val) / (max_val - min_val) * H
                        # unscaled boxes
                        z_min = min_val
                        z_q1  = q1
                        z_med = med
                        z_q3  = q3
                        z_max = max_val
                    
                    # Offset the boxplot in x so it doesn't overlap the node marker.
                    # x_box = x + dx
                    x_box = x
                    
                    # Create traces for each component of the boxplot:
                    trace_whisker_top = go.Scatter3d(
                        x=[x_box, x_box],
                        y=[y, y],
                        z=[z_q3, z_max],
                        mode='lines',
                        line=dict(color='grey', width=2),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    trace_whisker_bottom = go.Scatter3d(
                        x=[x_box, x_box],
                        y=[y, y],
                        z=[z_q1, z_min],
                        mode='lines',
                        line=dict(color='grey', width=2),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    trace_cap_top = go.Scatter3d(
                        x=[x_box - dx/2, x_box + dx/2],
                        y=[y, y],
                        z=[z_max, z_max],
                        mode='lines',
                        line=dict(color='grey', width=2),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    trace_cap_bottom = go.Scatter3d(
                        x=[x_box - dx/2, x_box + dx/2],
                        y=[y, y],
                        z=[z_min, z_min],
                        mode='lines',
                        line=dict(color='grey', width=2),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    # trace_box_left = go.Scatter3d(
                    #     x=[x_box - dx, x_box - dx],
                    #     y=[y, y],
                    #     z=[z_q1, z_q3],
                    #     mode='lines',
                    #     line=dict(color='black', width=4),
                    #     showlegend=False
                    # )
                    # trace_box_right = go.Scatter3d(
                    #     x=[x_box + dx, x_box + dx],
                    #     y=[y, y],
                    #     z=[z_q1, z_q3],
                    #     mode='lines',
                    #     line=dict(color='black', width=4),
                    #     showlegend=False
                    # )
                    trace_box = go.Scatter3d(
                        x=[x_box, x_box],
                        y=[y, y],
                        z=[z_q1, z_q3],
                        mode='lines',
                        line=dict(color='black', width=4),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    trace_medianx = go.Scatter3d(
                        x=[x_box - dx, x_box + dx],
                        y=[y, y],
                        z=[z_med, z_med],
                        mode='lines',
                        line=dict(color='red', width=3),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    trace_mediany = go.Scatter3d(
                        x=[x, x],
                        y=[y - dx, y + dx],
                        z=[z_med, z_med],
                        mode='lines',
                        line=dict(color='red', width=3),
                        opacity=opacity_noise_bar,
                        showlegend=False
                    )
                    # traces.extend([trace_whisker_top, trace_whisker_bottom,
                    #             trace_cap_top, trace_cap_bottom,
                    #             trace_box_left, trace_box_right,
                    #             trace_median])
                    traces.extend([trace_whisker_top, trace_whisker_bottom,
                                trace_cap_top, trace_cap_bottom,
                                trace_box,
                                trace_medianx, trace_mediany])
        
        print('Assigning camera and axes...')
        # Camera position
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        r = 2.5
        camera_eye = dict(
            x = r * np.cos(elevation) * np.cos(azimuth),
            y = r * np.cos(elevation) * np.sin(azimuth),
            z = r * np.sin(elevation)
        )
        # create substitue values for when custom axis range missing component
        if len(G.nodes) > 0:
            # Get values based on data
            x_values = [pos[node][0] for node in G.nodes()]
            y_values = [pos[node][1] for node in G.nodes()]
            fit_values = [data['fitness'] for _, data in G.nodes(data=True)]
            # Set range based on values
            x_min_sub, x_max_sub = min(x_values) - 1, max(x_values) + 1
            y_min_sub, y_max_sub = min(y_values) - 1, max(y_values) + 1
            z_min_sub, z_max_sub = min(fit_values) - 1, max(fit_values) + 1
            if node_noise:
                # If noise then include in range calculation
                z_max_sub = max(max(noisy_list) for noisy_list in node_noise.values()) + 1
                z_min_sub = min(min(noisy_list) for noisy_list in node_noise.values()) - 1
        else: # Default
            x_min_sub, x_max_sub, y_min_sub, y_max_sub, z_min_sub, z_max_sub = 1
        # Axis settings dicts
        xaxis_settings=dict(
            # title='X',
            title='',
            titlefont=dict(size=24, color='black'),
            tickfont=dict(size=16, color='black')
        )
        yaxis_settings=dict(
            # title='Y',
            title='',
            titlefont=dict(size=24, color='black'),
            tickfont=dict(size=16, color='black')
        )
        zaxis_settings=dict(
            title='Fitness',
            titlefont=dict(size=24, color='black'),  # Larger z-axis label
            tickfont=dict(size=16, color='black'),
        )
        # Apply custom axis options
        z_log_scale = False
        if z_log_scale == True:
            zaxis_settings['type'] = 'log'
        if axis_values.get("custom_x_min") is not None or axis_values.get("custom_x_max") is not None:
            custom_x_min = (
                axis_values.get("custom_x_min")
                if axis_values.get("custom_x_min") is not None
                else x_min_sub
            )
            custom_x_max = (
                axis_values.get("custom_x_max")
                if axis_values.get("custom_x_max") is not None
                else x_max_sub
            )
            xaxis_settings["range"] = [custom_x_min, custom_x_max]
        if axis_values.get("custom_y_min") is not None or axis_values.get("custom_y_max") is not None:
            custom_y_min = (
                axis_values.get("custom_y_min")
                if axis_values.get("custom_y_min") is not None
                else y_min_sub
            )
            custom_y_max = (
                axis_values.get("custom_y_max")
                if axis_values.get("custom_y_max") is not None
                else y_max_sub
            )
            yaxis_settings["range"] = [custom_y_min, custom_y_max]
        if axis_values.get("custom_z_min") is not None or axis_values.get("custom_z_max") is not None:
            custom_z_min = (
                axis_values.get("custom_z_min")
                if axis_values.get("custom_z_min") is not None
                else z_min_sub
            )
            custom_z_max = (
                axis_values.get("custom_z_max")
                if axis_values.get("custom_z_max") is not None
                else z_max_sub
            )
            zaxis_settings["range"] = [custom_z_min, custom_z_max]

        print('Displaying plot')
        # Create plot
        fig = go.Figure(data=traces)
        fig.update_layout(
        showlegend=False,
        width=1200,
        height=1200,
        scene=dict(
            camera=dict(
                eye=camera_eye
            ),
            xaxis=xaxis_settings,
            yaxis=yaxis_settings,
            zaxis=zaxis_settings
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
        )
    fig.write_html('plots/3dplot.html')
    return fig, debug_summary_component
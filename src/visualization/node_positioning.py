"""
Node positioning functions for the visualization module.

This module contains functions for calculating 2D positions of nodes
using various layout algorithms (MDS, t-SNE, Kamada-Kawai, spring layout).
"""

from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import TSNE

from ..common import hamming_distance, sol_tuple_ints, front_distance
from ..dashboard.DimensionalityReduction import landmark_mds, compute_distance_matrix


def _is_dual_front(G: nx.MultiDiGraph, noisy_nodes: List[Tuple[str, Dict]]) -> bool:
    """
    Check if we have dual-front mode (noisy front differs from base front).

    Args:
        G: NetworkX graph containing the nodes
        noisy_nodes: List of (node_name, node_data) tuples for noisy nodes

    Returns:
        True if any noisy front differs from its corresponding base front
    """
    for n_noisy, d_noisy in noisy_nodes:
        base = n_noisy.replace('_Noisy', '_True')
        if base in G.nodes:
            if G.nodes[base].get('front_solutions', []) != d_noisy.get('front_solutions', []):
                return True
    return False


def calculate_positions_mo(
    G: nx.MultiDiGraph,
    layout_type: str
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate 2D positions for multi-objective STN nodes.

    Uses front distance as the dissimilarity metric between nodes.

    Args:
        G: NetworkX graph containing MO nodes
        layout_type: Layout algorithm to use ('mds', 'tsne', 'kamada_kawai', etc.)

    Returns:
        Dictionary mapping node names to (x, y) coordinates
    """
    print('CALCULATING DISTANCES IN MULTIOBJECTIVE MODE')

    # Collect nodes
    base_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == 'STN_MO']
    noisy_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == 'STN_MO_Noise']

    # Decide mode: dual-front if any noisy front != its base front
    dual_front = _is_dual_front(G, noisy_nodes)

    # Nodes to embed
    embed_nodes = base_nodes + noisy_nodes if dual_front else base_nodes
    labels = [n for n, _ in embed_nodes]
    fronts = [d.get('front_solutions', []) for _, d in embed_nodes]
    K = len(fronts)

    pos = {}
    if K > 0:
        # Embedding
        if layout_type == 'r_lmds':
            # Landmark MDS - random landmarks
            print(f'\033[33mUsing Random Landmark MDS with {K} fronts\033[0m')
            XY = landmark_mds(front_distance, fronts, n_landmarks=None, random_state=42, landmark_method='random')
        elif layout_type == 'fps_lmds':
            # Landmark MDS - furthest point sampling
            print(f'\033[33mUsing FPS Landmark MDS with {K} fronts\033[0m')
            XY = landmark_mds(front_distance, fronts, n_landmarks=None, random_state=42, landmark_method='fps')
        elif layout_type == 'mds':
            # Standard MDS
            print(f'\033[33mUsing standard MDS with {K} fronts\033[0m')
            D = compute_distance_matrix(front_distance, fronts)
            mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42)
            XY = mds.fit_transform(D)
        elif layout_type == 'tsne':
            print(f'\033[33mUsing t-SNE with {K} fronts\033[0m')
            D = compute_distance_matrix(front_distance, fronts)
            tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
            XY = tsne.fit_transform(D)
        elif layout_type in ('kamada_kawai', 'kamada_kawai_weighted', 'spring'):
            print(f'\033[33mUsing Kamada-Kawai with {K} fronts\033[0m')
            D = compute_distance_matrix(front_distance, fronts)
            H = nx.complete_graph(K)
            for i in range(K):
                for j in range(i + 1, K):
                    H[i][j]['weight'] = max(D[i, j], 1e-6)
            raw = nx.kamada_kawai_layout(H, weight='weight', dim=2)
            XY = np.array([raw[i] for i in range(K)])
        else:
            # Default to MDS
            print(f'\033[33mUsing standard MDS (default) with {K} fronts\033[0m')
            D = compute_distance_matrix(front_distance, fronts)
            mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42)
            XY = mds.fit_transform(D)

        pos.update({labels[i]: (float(XY[i, 0]), float(XY[i, 1])) for i in range(K)})

    # If single-front mode, copy base XY to noisy nodes
    if not dual_front:
        for n_noisy, _ in noisy_nodes:
            base = n_noisy.replace('_Noisy', '_True')
            if base in pos:
                pos[n_noisy] = pos[base]

    return pos


def calculate_positions_so(
    G: nx.MultiDiGraph,
    layout_type: str,
    plot_3d: bool = False
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate 2D positions for single-objective nodes.

    Uses Hamming distance as the dissimilarity metric between solutions.

    Args:
        G: NetworkX graph containing nodes
        layout_type: Layout algorithm to use ('mds', 'tsne', 'kamada_kawai', etc.)
        plot_3d: Whether 3D plotting is enabled (affects some layout calculations)

    Returns:
        Dictionary mapping node names to (x, y) coordinates
    """
    print('\033[33mCompiling Solutions...\033[0m')

    # Collect unique solutions
    solutions_set = set()
    for node, data in G.nodes(data=True):
        sol = sol_tuple_ints(data.get('solution', []))
        if sol:  # Only add non-empty solutions
            solutions_set.add(sol)
    solutions_list = list(solutions_set)
    n = len(solutions_list)
    K = n

    if n == 0:
        print("ERROR: No solutions for Positioning")
        return {}

    pos = {}

    if layout_type == 'lmds':
        print(f'\033[33mUsing Landmark MDS with {K} solutions\033[0m')
        # Use Landmark MDS for efficient embedding
        positions_2d = landmark_mds(hamming_distance, solutions_list, n_landmarks=None, random_state=42)

        solution_positions = {}
        for i, sol in enumerate(solutions_list):
            solution_positions[sol] = positions_2d[i]

        for node, data in G.nodes(data=True):
            sol = tuple(data.get('solution', []))
            if sol in solution_positions:
                pos[node] = solution_positions[sol]

    elif layout_type == 'mds':
        print(f'\033[33mUsing standard MDS with {K} solutions\033[0m')
        dissimilarity_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                dissimilarity_matrix[i, j] = hamming_distance(solutions_list[i], solutions_list[j])

        mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42)
        positions_2d = mds.fit_transform(dissimilarity_matrix)

        solution_positions = {}
        for i, sol in enumerate(solutions_list):
            solution_positions[sol] = positions_2d[i]

        for node, data in G.nodes(data=True):
            sol = tuple(data.get('solution', []))
            if sol in solution_positions:
                pos[node] = solution_positions[sol]

    elif layout_type == 'tsne':
        print('\033[33mUsing TSNE\033[0m')
        dissimilarity_matrix = np.zeros((len(solutions_list), len(solutions_list)))
        for i in range(len(solutions_list)):
            for j in range(len(solutions_list)):
                dissimilarity_matrix[i, j] = hamming_distance(solutions_list[i], solutions_list[j])

        tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
        positions_2d = tsne.fit_transform(dissimilarity_matrix)

        solution_positions = {}
        for i, sol in enumerate(solutions_list):
            solution_positions[sol] = positions_2d[i]

        for node, data in G.nodes(data=True):
            sol = tuple(data.get('solution', []))
            if sol in solution_positions:
                pos[node] = solution_positions[sol]

    elif layout_type == 'kamada_kawai':
        print('\033[33mUsing Kamada Kawai\033[0m')
        # Calculate initial force-directed positions on the full graph G
        initial_pos = {}
        try:
            initial_pos = nx.kamada_kawai_layout(G, dim=2, scale=1)
            print(f"Kamada-Kawai initial layout calculated for {len(initial_pos)} nodes.")
        except Exception as e:
            print(f"Kamada-Kawai layout on full graph G failed: {e}")
            print("Falling back to random positions.")
            pos = {node: (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1) for node in G.nodes()}
            initial_pos = None

        # Proceed with averaging only if initial_pos was successfully calculated
        if initial_pos:
            # Group nodes by solution and collect their initial positions
            positions_by_solution = {}
            nodes_without_solution_or_pos = []

            for node, data in G.nodes(data=True):
                node_pos = initial_pos.get(node)
                if node_pos is None:
                    print(f"Warning: Node {node} missing from initial KK position results.")
                    nodes_without_solution_or_pos.append(node)
                    continue

                sol_data = data.get('solution')
                if sol_data is not None:
                    try:
                        sol_tuple = sol_tuple_ints(sol_data)
                        if sol_tuple not in positions_by_solution:
                            positions_by_solution[sol_tuple] = []
                        positions_by_solution[sol_tuple].append(node_pos)
                    except Exception as e_conv:
                        print(f"Warning: Could not process solution for node {node}: {e_conv}.")
                        nodes_without_solution_or_pos.append(node)
                else:
                    nodes_without_solution_or_pos.append(node)

            print(f"Processed {len(initial_pos)} nodes. Found {len(positions_by_solution)} unique solutions with positions.")
            if nodes_without_solution_or_pos:
                print(f"Found {len(nodes_without_solution_or_pos)} nodes without a 'solution' attribute or missing from initial positions.")

            # Calculate the average position for each unique solution
            final_solution_positions = {}
            for sol_tuple, pos_list in positions_by_solution.items():
                if not pos_list:
                    continue
                avg_pos = np.mean(np.array(pos_list), axis=0)
                final_solution_positions[sol_tuple] = tuple(avg_pos)

            # Assign the final (averaged) position to all nodes in G
            assigned_count = 0
            unassigned_count = 0
            for node, data in G.nodes(data=True):
                sol_data = data.get('solution')
                assigned = False
                if sol_data is not None:
                    try:
                        sol_tuple = sol_tuple_ints(sol_data)
                        if sol_tuple in final_solution_positions:
                            pos[node] = final_solution_positions[sol_tuple]
                            assigned = True
                            assigned_count += 1
                    except Exception:
                        pass

                if not assigned:
                    pos[node] = initial_pos.get(node, (np.random.rand() * 0.1, np.random.rand() * 0.1))
                    unassigned_count += 1

            print(f"Assigned final positions to {assigned_count} nodes based on averaged solution positions.")
            if unassigned_count > 0:
                print(f"Assigned initial/fallback positions to {unassigned_count} nodes (no solution/pos issue).")
        elif 'pos' not in locals() or not pos:
            pos = {node: (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1) for node in G.nodes()}

    elif layout_type == 'kamada_kawai_weighted':
        print('\033[33mUsing Kamada Kawai Weighted\033[0m')
        # Build a complete graph of unique solutions
        CG = nx.complete_graph(n)
        mapping = {i: solutions_list[i] for i in range(n)}
        # For each pair, set the edge weight to be the Hamming distance
        for i in range(n):
            for j in range(i + 1, n):
                weight = hamming_distance(solutions_list[i], solutions_list[j])
                CG[i][j]['weight'] = weight

        # Compute the Kamada-Kawai layout on H using the weight attribute
        pos_unique = nx.kamada_kawai_layout(CG, weight='weight', dim=2)

        # Map unique solution positions back to a dictionary keyed by the actual solution tuple
        solution_positions = {mapping[i]: pos_unique[i] for i in range(n)}

        # For every node in G, assign the position corresponding to its solution
        for node, data in G.nodes(data=True):
            sol = tuple(data.get('solution', []))
            if sol in solution_positions:
                pos[node] = solution_positions[sol]

    else:
        # Default: spring layout
        print('\033[33mUsing Spring Layout\033[0m')
        pos = nx.spring_layout(G, dim=2 if not plot_3d else 3)
        # Update positions for noisy nodes
        for node in G.nodes():
            if node.startswith("Noisy_"):
                solution_node = node.replace("Noisy_", "", 1)
                if solution_node in pos:
                    pos[node] = pos[solution_node]

    return pos


def calculate_positions(
    G: nx.MultiDiGraph,
    layout_type: str,
    stn_plot_type: str = 'posterior',
    plot_3d: bool = False
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate 2D positions for all nodes in the graph.

    Routes to the appropriate positioning function based on the STN plot type.

    Args:
        G: NetworkX graph containing nodes
        layout_type: Layout algorithm to use
        stn_plot_type: STN plot type ('posterior', 'prior', 'multiobjective')
        plot_3d: Whether 3D plotting is enabled

    Returns:
        Dictionary mapping node names to (x, y) coordinates
    """
    print('\033[33mCalculating node positions...\033[0m')

    if stn_plot_type == 'multiobjective':
        pos = calculate_positions_mo(G, layout_type)
    else:
        # Both 'posterior' and 'prior' use SO positioning (Hamming distance)
        pos = calculate_positions_so(G, layout_type, plot_3d)

    print('\033[32mNode Positions Calculated\033[0m')
    return pos


def create_hover_text(
    G: nx.MultiDiGraph,
    hover_info_value: str
) -> List[str]:
    """
    Create hover text for nodes based on the selected info type.

    Args:
        G: NetworkX graph containing nodes
        hover_info_value: Type of info to show ('fitness', 'iterations', 'solutions')

    Returns:
        List of hover text strings in node order
    """
    node_hover_text = []
    if hover_info_value == 'fitness':
        node_hover_text = [str(G.nodes[node].get('fitness', '')) for node in G.nodes()]
    elif hover_info_value == 'iterations':
        node_hover_text = [str(G.nodes[node].get('iterations', '')) for node in G.nodes()]
    elif hover_info_value == 'solutions':
        node_hover_text = [str(G.nodes[node].get('solution', '')) for node in G.nodes()]
    return node_hover_text

"""
Graph building functions for the visualization module.

This module contains functions for constructing NetworkX graphs from
trajectory data, local optima networks, and multi-objective fronts.
Functions take a graph and modify it in place, returning any mappings needed.
"""

from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np

from .config import PlotConfig
from ..common import hamming_distance, sol_tuple_ints, lookup_map


def generate_run_summary_string(selected_trajectories: List) -> str:
    """
    Generate a debug summary string for a set of trajectories.

    Args:
        selected_trajectories: List of trajectory entries, each containing
            (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)

    Returns:
        A formatted string summarizing the runs.
    """
    lines = []
    for run_idx, entry in enumerate(selected_trajectories):
        if len(entry) < 5:
            lines.append(f"Skipping malformed entry in run {run_idx}")
            continue
        unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry[:5]
        # Convert noisy fitnesses to ints if needed
        noisy_fitnesses = [int(fit) for fit in noisy_fitnesses]
        lines.append(f"Run {run_idx}:")
        for i, solution in enumerate(unique_solutions):
            lines.append(f"  Solution: {solution} | Fitness: {unique_fitnesses[i]} | Noisy Fitness: {noisy_fitnesses[i]}")
        lines.append("")  # Blank line between runs
    return "\n".join(lines)


def print_hamming_transitions(
    all_run_trajectories: List,
    print_sols: bool = False,
    print_transitions: bool = False
) -> None:
    """
    Print normalized Hamming distances between consecutive solutions for each run.

    Args:
        all_run_trajectories: List of runs, where each run is a tuple of
            (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)
        print_sols: If True, print the actual solution values
        print_transitions: If True, print transition labels
    """
    overall_distances = []

    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 5:
            print(f"Run {run_idx}: Skipping malformed entry (expected at least 5 elements, got {len(entry)})")
            continue

        unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry[:5]
        distances = []

        print(f"Run {run_idx}:")
        for i in range(len(unique_solutions) - 1):
            sol1 = unique_solutions[i]
            sol2 = unique_solutions[i + 1]
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


def add_stn_trajectories(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    algo_idx: int,
    stn_node_mapping: Dict[Tuple, str],
    config: PlotConfig
) -> Dict[Tuple, str]:
    """
    Add STN (Search Trajectory Network) nodes and edges to the graph.

    Creates nodes for each unique solution encountered in the trajectories,
    and edges for each transition between solutions.

    Args:
        G: NetworkX MultiDiGraph to modify
        all_run_trajectories: List of trajectory runs
        edge_color: Color to use for edges from this algorithm
        algo_idx: Index of the algorithm (for labeling)
        stn_node_mapping: Existing mapping of (solution, type) -> node_label
        config: PlotConfig object with settings

    Returns:
        Updated stn_node_mapping dictionary
    """
    lower_fit_limit = config.stn.lower_fit_limit
    edge_size = config.stn.edge_size

    for run_idx, entry in enumerate(all_run_trajectories):
        # Check data length and None values (5 core elements + optional noisy variant data)
        if len(entry) < 5:
            print(f"Skipping malformed entry {entry}, expected at least 5 elements but got {len(entry)}")
            continue
        unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry[:5]
        noisy_fitnesses = [int(fit) for fit in noisy_fitnesses]
        if any(x is None for x in (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)):
            print(f"Skipping run {run_idx} due to None values: {entry}")
            continue

        # Create nodes and store node labels in order for this run
        for i, solution in enumerate(unique_solutions):
            current_fitness = unique_fitnesses[i]
            if lower_fit_limit is not None:
                if current_fitness < lower_fit_limit:
                    # Skip adding this node because its fitness is below the threshold
                    continue
            start_node = True if i == 0 else False
            end_node = True if i == len(unique_solutions) - 1 else False
            solution_tuple = tuple(solution)
            key = (solution_tuple, "STN")
            if key not in stn_node_mapping:
                node_label = f"STN_{len(stn_node_mapping) + 1}"
                stn_node_mapping[key] = node_label
                G.add_node(
                    node_label,
                    solution=solution,
                    fitness=unique_fitnesses[i],
                    iterations=solution_iterations[i],
                    type="STN",
                    run_idx=run_idx,
                    step=i,
                    color=edge_color,
                    start_node=start_node,
                    end_node=end_node
                )
            else:
                node_label = stn_node_mapping[key]
                print(f"DEBUG: Reusing STN node {node_label} for solution {solution_tuple}")

            # Add noisy node for STN data (if desired)
            noisy_node_label = f"Noisy_{node_label}"
            if noisy_node_label not in G.nodes():
                try:
                    G.add_node(noisy_node_label, solution=solution, fitness=noisy_fitnesses[i], color=edge_color)
                except Exception as e:
                    print(f"Error adding noisy node: {noisy_node_label}, {e}")
                G.add_edge(
                    node_label,
                    noisy_node_label,
                    weight=edge_size,
                    color=edge_color,
                    edge_type='Noise'
                )

        # Add transitions as STN edges
        for j, (prev_solution, current_solution) in enumerate(transitions):
            prev_key = (tuple(prev_solution), "STN")
            curr_key = (tuple(current_solution), "STN")
            if prev_key in stn_node_mapping and curr_key in stn_node_mapping:
                src = stn_node_mapping[prev_key]
                tgt = stn_node_mapping[curr_key]
                G.add_edge(src, tgt, weight=edge_size, color=edge_color, edge_type='STN')

    return stn_node_mapping


def add_mo_fronts(
    G: nx.MultiDiGraph,
    mo_runs_for_series: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey'
) -> List[Tuple[str, Dict]]:
    """
    Add multi-objective STN nodes and edges to the graph.

    Creates nodes for each front at each generation, with optional noisy nodes
    when dual metrics are provided. Connects consecutive generations with edges.

    Args:
        G: NetworkX MultiDiGraph to modify
        mo_runs_for_series: List of runs, each containing a sequence of front data
        edge_color: Color to use for edges
        series_idx: Index of the series (for labeling)
        noisy_node_color: Color for noisy/noise edges

    Returns:
        List of (node_name, node_data) tuples for all MO nodes added
    """
    print(f"[MO DEBUG] add_mo_fronts_to_graph(series_idx={series_idx})", flush=True)
    mo_nodes = []

    for run_idx, front_seq in enumerate(mo_runs_for_series or []):
        prev_base = None

        count_front1 = sum(1 for it in front_seq if (it.get('front1') is not None))
        count_front2 = sum(1 for it in front_seq if (it.get('front2') is not None))
        print(f"[MO DEBUG] Run {run_idx}: front1_gens={count_front1}, front2_gens={count_front2}, total_gens={len(front_seq)}", flush=True)

        for item in front_seq or []:
            front1 = item.get('front1') or []
            front2 = item.get('front2') or None
            m1_raw = item.get('metric1', 0.0)
            m2_raw = item.get('metric2', None)
            g = int(item.get('gen_idx', 0))

            # Safe numeric conversion
            try:
                m1 = float(m1_raw if m1_raw is not None else 0.0)
            except Exception:
                m1 = 0.0
            try:
                m2 = float(m2_raw) if m2_raw is not None else None
            except Exception:
                m2 = None

            # -------- base (front1, metric1) node --------
            node_base = f"MO_S{series_idx}_R{run_idx}_G{g}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_MO",
                    is_noisy=False,
                    front_solutions=front1,
                    front_size=len(front1),
                    hypervolume=m1,
                    fitness=m1,  # z-axis value
                    run_idx=run_idx,
                    gen_idx=g,
                    color=edge_color,
                )
            mo_nodes.append((node_base, G.nodes[node_base]))

            # -------- optional noisy node (metric2) --------
            if m2 is not None:
                noisy_front = front2 if (front2 is not None) else front1
                node_noisy = f"MO_S{series_idx}_R{run_idx}_G{g}_Noisy"
                if node_noisy not in G.nodes:
                    G.add_node(
                        node_noisy,
                        type="STN_MO_Noise",
                        is_noisy=True,
                        front_solutions=noisy_front,
                        front_size=len(noisy_front),
                        hypervolume=m2,
                        fitness=m2,
                        run_idx=run_idx,
                        gen_idx=g,
                        color=edge_color,
                    )
                mo_nodes.append((node_noisy, G.nodes[node_noisy]))

                # edge between metric1 and metric2 nodes (noisy)
                G.add_edge(
                    node_base,
                    node_noisy,
                    weight=0.5,
                    color=noisy_node_color,
                    edge_type="Noise_MO",
                    is_noisy=True,
                )

            # -------- temporal link across generations --------
            if prev_base is not None:
                G.add_edge(
                    prev_base,
                    node_base,
                    weight=0.5,
                    color=edge_color,
                    edge_type="STN_MO",
                    is_noisy=False,
                )
            prev_base = node_base

    return mo_nodes


def add_prior_noise_stn(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey',
    dedup: bool = False
) -> None:
    """
    Add single-objective STN nodes with prior noise variants to the graph.

    Creates base nodes for each unique solution and separate noisy nodes
    for each variant in noisy_sol_variants, connected by noise edges.
    Consecutive base nodes are connected by temporal edges.

    Args:
        G: NetworkX MultiDiGraph to modify
        all_run_trajectories: List of runs, each a 7-element list:
            [unique_sols, unique_fits, noisy_fits, sol_iterations,
             sol_transitions, noisy_sol_variants, noisy_variant_fitnesses]
        edge_color: Color for temporal edges (algorithm color)
        series_idx: Index of the algorithm series
        noisy_node_color: Color for noise edges
        dedup: If True, deduplicate noisy variants per solution (keep first occurrence)
    """
    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 7:
            print(f"[Prior noise] Skipping entry with {len(entry)} elements, expected 7")
            continue

        unique_sols, unique_fits, noisy_fits, sol_iterations, transitions, \
            noisy_sol_variants, noisy_variant_fitnesses = entry[:7]

        if unique_sols is None:
            continue

        prev_base = None

        for i, solution in enumerate(unique_sols):
            true_fitness = unique_fits[i]

            # -------- base node --------
            node_base = f"STN_S{series_idx}_R{run_idx}_Sol{i}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_SO",
                    is_noisy=False,
                    solution=solution,
                    fitness=true_fitness,
                    iterations=sol_iterations[i] if i < len(sol_iterations) else 1,
                    sol_idx=i,
                    run_idx=run_idx,
                    series_idx=series_idx,
                    color=edge_color,
                )

            # -------- noisy variant nodes --------
            variants = noisy_sol_variants[i] if i < len(noisy_sol_variants) else []
            variant_fits = noisy_variant_fitnesses[i] if i < len(noisy_variant_fitnesses) else []

            # Optionally deduplicate variants by solution tuple
            if dedup and variants:
                seen = set()
                deduped_variants = []
                deduped_fits = []
                for j, v_sol in enumerate(variants):
                    v_key = tuple(v_sol)
                    if v_key not in seen:
                        seen.add(v_key)
                        deduped_variants.append(v_sol)
                        deduped_fits.append(variant_fits[j] if j < len(variant_fits) else 0.0)
                variants = deduped_variants
                variant_fits = deduped_fits

            for j, variant_sol in enumerate(variants):
                variant_fitness = variant_fits[j] if j < len(variant_fits) else 0.0
                node_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_Var{j}_Noisy"
                if node_noisy not in G.nodes:
                    G.add_node(
                        node_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=variant_sol,
                        fitness=variant_fitness,
                        sol_idx=i,
                        var_idx=j,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )

                    # noise edge: base -> noisy variant
                    G.add_edge(
                        node_base,
                        node_noisy,
                        weight=0.5,
                        color=noisy_node_color,
                        edge_type="Noise_SO",
                        is_noisy=True,
                    )

            # -------- temporal edge across solutions --------
            if prev_base is not None:
                G.add_edge(
                    prev_base,
                    node_base,
                    weight=0.5,
                    color=edge_color,
                    edge_type="STN_SO",
                    is_noisy=False,
                )
            prev_base = node_base


def add_prior_noise_stn_v2(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey',
    dedup: bool = False
) -> None:
    """
    Add single-objective STN nodes with two kinds of noisy variant nodes.

    For each unique solution (base node with true fitness) this creates:
      1) "Solution-noisy" nodes: use noisy_sol_variants (perturbed solutions)
         but keep the *true* fitness of the base node, so they sit on the
         same fitness plane.  Positioned by Hamming distance from their
         perturbed solution.
      2) "Fitness-noisy" nodes: use the *original* (unperturbed) solution
         but with the noisy fitness values from unique_noisy_fits, so they
         differ only in height (z-axis).
    Both kinds are connected to the base node via noise edges.
    Consecutive base nodes are connected by temporal edges.

    Args:
        G: NetworkX MultiDiGraph to modify
        all_run_trajectories: List of runs, each a 7-element list:
            [unique_sols, unique_fits, noisy_fits, sol_iterations,
             sol_transitions, noisy_sol_variants, noisy_variant_fitnesses]
        edge_color: Color for temporal edges (algorithm color)
        series_idx: Index of the algorithm series
        noisy_node_color: Color for noise edges
        dedup: If True, deduplicate solution-noisy variants per base node
    """
    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 7:
            print(f"[Prior noise V2] Skipping entry with {len(entry)} elements, expected 7")
            continue

        unique_sols, unique_fits, noisy_fits, sol_iterations, transitions, \
            noisy_sol_variants, noisy_variant_fitnesses = entry[:7]

        if unique_sols is None:
            continue

        prev_base = None

        for i, solution in enumerate(unique_sols):
            true_fitness = unique_fits[i]

            # -------- base node --------
            node_base = f"STN_S{series_idx}_R{run_idx}_Sol{i}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_SO",
                    is_noisy=False,
                    solution=solution,
                    fitness=true_fitness,
                    iterations=sol_iterations[i] if i < len(sol_iterations) else 1,
                    sol_idx=i,
                    run_idx=run_idx,
                    series_idx=series_idx,
                    color=edge_color,
                )

            # -------- Type 1: solution-noisy nodes --------
            # Perturbed solutions but with the TRUE fitness of the base node
            variants = noisy_sol_variants[i] if i < len(noisy_sol_variants) else []

            if dedup and variants:
                seen = set()
                deduped_variants = []
                for v_sol in variants:
                    v_key = tuple(v_sol)
                    if v_key not in seen:
                        seen.add(v_key)
                        deduped_variants.append(v_sol)
                variants = deduped_variants

            for j, variant_sol in enumerate(variants):
                node_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_SolVar{j}_Noisy"
                if node_noisy not in G.nodes:
                    G.add_node(
                        node_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=variant_sol,
                        fitness=true_fitness,  # same fitness as base
                        sol_idx=i,
                        var_idx=j,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- Type 2: fitness-noisy nodes --------
            # Same solution as base but with noisy fitness values
            noisy_fit_list = noisy_fits[i] if i < len(noisy_fits) else []
            if not isinstance(noisy_fit_list, (list, tuple)):
                noisy_fit_list = [noisy_fit_list]

            for k, nf in enumerate(noisy_fit_list):
                node_fit_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_FitVar{k}_Noisy"
                if node_fit_noisy not in G.nodes:
                    G.add_node(
                        node_fit_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=solution,  # same solution as base
                        fitness=nf,         # noisy fitness
                        sol_idx=i,
                        var_idx=k,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_fit_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- temporal edge --------
            if prev_base is not None:
                G.add_edge(
                    prev_base, node_base,
                    weight=0.5, color=edge_color,
                    edge_type="STN_SO", is_noisy=False,
                )
            prev_base = node_base


def add_prior_noise_stn_v3(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey',
    dedup: bool = False
) -> None:
    """
    Add single-objective STN nodes with two kinds of noisy variant nodes (V3).

    Same as V2 except the fitness-noisy nodes use noisy_variant_fitnesses
    (multiple values per solution from all evaluations) instead of
    unique_noisy_fits (single value per solution).

    For each unique solution (base node with true fitness) this creates:
      1) "Solution-noisy" nodes: use noisy_sol_variants (perturbed solutions)
         but keep the *true* fitness of the base node.
      2) "Fitness-noisy" nodes: use the *original* (unperturbed) solution
         but with noisy_variant_fitnesses (one node per evaluation).
    Both kinds are connected to the base node via noise edges.
    Consecutive base nodes are connected by temporal edges.

    Args:
        G: NetworkX MultiDiGraph to modify
        all_run_trajectories: List of runs, each a 7-element list:
            [unique_sols, unique_fits, noisy_fits, sol_iterations,
             sol_transitions, noisy_sol_variants, noisy_variant_fitnesses]
        edge_color: Color for temporal edges (algorithm color)
        series_idx: Index of the algorithm series
        noisy_node_color: Color for noise edges
        dedup: If True, deduplicate solution-noisy variants per base node
    """
    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 7:
            print(f"[Prior noise V3] Skipping entry with {len(entry)} elements, expected 7")
            continue

        unique_sols, unique_fits, noisy_fits, sol_iterations, transitions, \
            noisy_sol_variants, noisy_variant_fitnesses = entry[:7]

        if unique_sols is None:
            continue

        prev_base = None

        for i, solution in enumerate(unique_sols):
            true_fitness = unique_fits[i]

            # -------- base node --------
            node_base = f"STN_S{series_idx}_R{run_idx}_Sol{i}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_SO",
                    is_noisy=False,
                    solution=solution,
                    fitness=true_fitness,
                    iterations=sol_iterations[i] if i < len(sol_iterations) else 1,
                    sol_idx=i,
                    run_idx=run_idx,
                    series_idx=series_idx,
                    color=edge_color,
                )

            # -------- Type 1: solution-noisy nodes --------
            # Perturbed solutions but with the TRUE fitness of the base node
            variants = noisy_sol_variants[i] if i < len(noisy_sol_variants) else []

            if dedup and variants:
                seen = set()
                deduped_variants = []
                for v_sol in variants:
                    v_key = tuple(v_sol)
                    if v_key not in seen:
                        seen.add(v_key)
                        deduped_variants.append(v_sol)
                variants = deduped_variants

            for j, variant_sol in enumerate(variants):
                node_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_SolVar{j}_Noisy"
                if node_noisy not in G.nodes:
                    G.add_node(
                        node_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=variant_sol,
                        fitness=true_fitness,  # same fitness as base
                        sol_idx=i,
                        var_idx=j,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- Type 2: fitness-noisy nodes --------
            # Same solution as base but with noisy_variant_fitnesses (multiple per solution)
            variant_fits = noisy_variant_fitnesses[i] if i < len(noisy_variant_fitnesses) else []
            if not isinstance(variant_fits, (list, tuple)):
                variant_fits = [variant_fits]

            if dedup and variant_fits:
                variant_fits = list(set(variant_fits))

            for k, nf in enumerate(variant_fits):
                node_fit_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_FitVar{k}_Noisy"
                if node_fit_noisy not in G.nodes:
                    G.add_node(
                        node_fit_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=solution,  # same solution as base
                        fitness=nf,         # noisy fitness from variant evaluations
                        sol_idx=i,
                        var_idx=k,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_fit_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- temporal edge --------
            if prev_base is not None:
                G.add_edge(
                    prev_base, node_base,
                    weight=0.5, color=edge_color,
                    edge_type="STN_SO", is_noisy=False,
                )
            prev_base = node_base


def add_prior_noise_stn_v4(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey',
    dedup: bool = False
) -> None:
    """
    Add single-objective STN nodes with two kinds of noisy nodes (V4).

    For each unique solution (base node: true solution + true fitness):
      1) "Fitness-noisy" node: same solution as base, but noisy fitness
         (from unique_noisy_fits). Differs only in z-axis.
      2) "Solution-noisy" node: noisy (perturbed) solution from
         unique_noisy_sols, but same true fitness as base.
         Differs in x/y position (Hamming distance) but same z.
    Both connected to base via noise edges.
    Consecutive base nodes connected by temporal edges.
    Dedup applies to solution-noisy nodes (from noisy_sol_variants).

    Entry format (8 elements):
        [unique_sols, unique_fits, noisy_fits, sol_iterations,
         sol_transitions, noisy_sol_variants, noisy_variant_fitnesses,
         unique_noisy_sols]
    """
    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 8:
            print(f"[Prior noise V4] Skipping entry with {len(entry)} elements, expected 8")
            continue

        unique_sols, unique_fits, noisy_fits, sol_iterations, transitions, \
            noisy_sol_variants, noisy_variant_fitnesses, unique_noisy_sols = entry

        if unique_sols is None:
            continue

        prev_base = None

        for i, solution in enumerate(unique_sols):
            true_fitness = unique_fits[i]

            # -------- base node --------
            node_base = f"STN_S{series_idx}_R{run_idx}_Sol{i}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_SO",
                    is_noisy=False,
                    solution=solution,
                    fitness=true_fitness,
                    iterations=sol_iterations[i] if i < len(sol_iterations) else 1,
                    sol_idx=i,
                    run_idx=run_idx,
                    series_idx=series_idx,
                    color=edge_color,
                )

            # -------- Type 1: fitness-noisy node --------
            # Same solution as base, noisy fitness
            noisy_fit = noisy_fits[i] if i < len(noisy_fits) else None
            if noisy_fit is not None:
                node_fit_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_FitNoisy"
                if node_fit_noisy not in G.nodes:
                    G.add_node(
                        node_fit_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=solution,
                        fitness=noisy_fit,
                        sol_idx=i,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_fit_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- Type 2: solution-noisy node --------
            # Noisy (perturbed) solution, same true fitness as base
            noisy_sol = unique_noisy_sols[i] if i < len(unique_noisy_sols) else None
            if noisy_sol is not None:
                node_sol_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_SolNoisy"
                if node_sol_noisy not in G.nodes:
                    G.add_node(
                        node_sol_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=noisy_sol,
                        fitness=true_fitness,
                        sol_idx=i,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_sol_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- temporal edge --------
            if prev_base is not None:
                G.add_edge(
                    prev_base, node_base,
                    weight=0.5, color=edge_color,
                    edge_type="STN_SO", is_noisy=False,
                )
            prev_base = node_base


def add_prior_noise_stn_v5(
    G: nx.MultiDiGraph,
    all_run_trajectories: List,
    edge_color: str,
    series_idx: int,
    noisy_node_color: str = 'grey',
    dedup: bool = False
) -> None:
    """
    Add single-objective STN nodes with one noisy node per base (V5).

    For each unique solution (base node: true solution + true fitness):
      1) One noisy node using the noisy solution and noisy fitness.
    Connected to base via a noise edge.
    Consecutive base nodes connected by temporal edges.

    Entry format (8 elements):
        [unique_sols, unique_fits, noisy_fits, sol_iterations,
         sol_transitions, noisy_sol_variants, noisy_variant_fitnesses,
         unique_noisy_sols]
    """
    for run_idx, entry in enumerate(all_run_trajectories):
        if len(entry) < 8:
            print(f"[Prior noise V5] Skipping entry with {len(entry)} elements, expected 8")
            continue

        unique_sols, unique_fits, noisy_fits, sol_iterations, transitions, \
            noisy_sol_variants, noisy_variant_fitnesses, unique_noisy_sols = entry

        if unique_sols is None:
            continue

        prev_base = None

        for i, solution in enumerate(unique_sols):
            true_fitness = unique_fits[i]

            # -------- base node --------
            node_base = f"STN_S{series_idx}_R{run_idx}_Sol{i}_True"
            if node_base not in G.nodes:
                G.add_node(
                    node_base,
                    type="STN_SO",
                    is_noisy=False,
                    solution=solution,
                    fitness=true_fitness,
                    iterations=sol_iterations[i] if i < len(sol_iterations) else 1,
                    sol_idx=i,
                    run_idx=run_idx,
                    series_idx=series_idx,
                    color=edge_color,
                )

            # -------- noisy node: noisy solution + noisy fitness --------
            noisy_sol = unique_noisy_sols[i] if i < len(unique_noisy_sols) else None
            noisy_fit = noisy_fits[i] if i < len(noisy_fits) else None
            if noisy_sol is not None and noisy_fit is not None:
                node_noisy = f"STN_S{series_idx}_R{run_idx}_Sol{i}_Noisy"
                if node_noisy not in G.nodes:
                    G.add_node(
                        node_noisy,
                        type="STN_SO_Noise",
                        is_noisy=True,
                        solution=noisy_sol,
                        fitness=noisy_fit,
                        sol_idx=i,
                        run_idx=run_idx,
                        series_idx=series_idx,
                        color=edge_color,
                    )
                    G.add_edge(
                        node_base, node_noisy,
                        weight=0.5, color=noisy_node_color,
                        edge_type="Noise_SO", is_noisy=True,
                    )

            # -------- temporal edge --------
            if prev_base is not None:
                G.add_edge(
                    prev_base, node_base,
                    weight=0.5, color=edge_color,
                    edge_type="STN_SO", is_noisy=False,
                )
            prev_base = node_base


def add_lon_nodes(
    G: nx.MultiDiGraph,
    local_optima: Dict,
    lon_node_mapping: Dict[Tuple, str],
    config: PlotConfig,
    problem_id: str,
    fitness_func_params: Optional[Dict] = None
) -> Tuple[Dict[Tuple, str], Dict[str, List[float]]]:
    """
    Add LON (Local Optima Network) nodes to the graph.

    Creates nodes for each local optimum and optionally computes noisy
    fitness samples for each node.

    Args:
        G: NetworkX MultiDiGraph to modify
        local_optima: Dictionary containing local_optima, fitness_values, and edges
        lon_node_mapping: Existing mapping of (solution, type) -> node_label
        config: PlotConfig object with settings
        problem_id: Problem identifier for loading problem data
        fitness_func_params: Optional dict with items_dict, capacity, etc.

    Returns:
        Tuple of (updated lon_node_mapping, node_noise dictionary)
    """
    from ..problems.FitnessFunctions import (
        eval_noisy_kp_v1_simple, eval_noisy_kp_v2_simple,
        eval_noisy_kp_v1, eval_noisy_kp_v2, eval_noisy_kp_prior
    )
    from ..problems.ProblemScripts import load_problem_KP

    node_noise = {}

    for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
        solution_tuple = tuple(opt)
        key = (solution_tuple, "LON")
        if key not in lon_node_mapping:
            node_label = f"Local Optimum {len(lon_node_mapping) + 1}"
            lon_node_mapping[key] = node_label
            G.add_node(node_label, solution=opt, fitness=fitness, type="LON")
        else:
            node_label = lon_node_mapping[key]

        # NOISE BOX PLOTS FOR LON
        node_noise[node_label] = []
        n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(problem_id)

        nlon_config = config.noisy_lon
        for i in range(nlon_config.samples):
            # Compute the noisy fitness
            if nlon_config.fit_func == 'kpv1s':
                noisy_fitness = eval_noisy_kp_v1_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv2s':
                noisy_fitness = eval_noisy_kp_v2_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv1mw':
                noisy_fitness = eval_noisy_kp_v1(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv2mw':
                noisy_fitness = eval_noisy_kp_v2(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpp':
                noisy_fitness, _ = eval_noisy_kp_prior(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            else:
                print('NO NOISY FITNESS FUNCTION PROVIDED')
                noisy_fitness = 0
            node_noise[node_label].append(noisy_fitness)

    return lon_node_mapping, node_noise


def add_lon_edges(
    G: nx.MultiDiGraph,
    local_optima: Dict,
    lon_node_mapping: Dict[Tuple, str],
    config: PlotConfig,
    opt_feas_map: Optional[Dict] = None
) -> None:
    """
    Add LON edges to the graph.

    Creates edges between local optima nodes based on the transition data,
    with optional coloring based on feasibility.

    Args:
        G: NetworkX MultiDiGraph to modify
        local_optima: Dictionary containing edges data
        lon_node_mapping: Mapping of (solution, type) -> node_label
        config: PlotConfig object with settings
        opt_feas_map: Optional feasibility map for edge coloring
    """
    import plotly.express as px
    # lookup_map imported at module level from common

    colour_edges_by_feas = config.lon.edge_colour_feas

    for (source, target), weight in local_optima["edges"].items():
        source_tuple = tuple(source)
        target_tuple = tuple(target)
        src_key = (source_tuple, "LON")
        tgt_key = (target_tuple, "LON")
        if src_key in lon_node_mapping and tgt_key in lon_node_mapping:
            src_label = lon_node_mapping[src_key]
            tgt_label = lon_node_mapping[tgt_key]

            edge_color = 'black'  # default
            if colour_edges_by_feas and opt_feas_map:
                tgt_sol = G.nodes[tgt_label].get('solution', [])
                feas = lookup_map(opt_feas_map, tgt_sol)
                if feas is not None:
                    edge_color = 'green' if int(feas) == 1 else 'red'

            G.add_edge(src_label, tgt_label, weight=weight, color=edge_color, edge_type='LON')

    # ONLY recolor by weight if we're NOT colouring by feasibility
    if not colour_edges_by_feas:
        # Calculate min and max edge weight for LON for normalisation
        LON_edge_weight_all = [
            data.get('weight', 2)
            for u, v, key, data in G.edges(data=True, keys=True)
            if "Local Optimum" in u and "Local Optimum" in v
        ]
        if LON_edge_weight_all:
            LON_edge_weight_min = min(LON_edge_weight_all)
            LON_edge_weight_max = max(LON_edge_weight_all)
        else:
            LON_edge_weight_min = LON_edge_weight_max = 1

        # Normalise edge weights for edges between Local Optimum nodes and colour
        for u, v, key, data in G.edges(data=True, keys=True):
            if "Local Optimum" in u and "Local Optimum" in v:
                weight = data.get('weight', 2)
                # Normalize the weight (if all weights are equal, default to 0.5)
                norm_weight = (weight - LON_edge_weight_min) / (LON_edge_weight_max - LON_edge_weight_min) if LON_edge_weight_max > LON_edge_weight_min else 0.5
                norm_weight = np.clip(norm_weight, 0, 0.9999)
                color = px.colors.sample_colorscale('plasma', norm_weight)[0]
                data['norm_weight'] = norm_weight
                data['color'] = color


def debug_mo_counts(
    G: nx.MultiDiGraph,
    by: str = "run_idx",
    label: str = "[MO]",
    list_fronts: bool = False,
    max_list: int = 12
) -> str:
    """
    Generate a summary of MO (multi-objective) nodes and edges in the graph.

    Args:
        G: NetworkX MultiDiGraph to analyze
        by: Attribute to group by ('run_idx' or 'series_idx')
        label: Label prefix for output
        list_fronts: If True, list front sizes per generation
        max_list: Maximum number of entries to list

    Returns:
        Summary string (also printed to console)
    """
    # All MO nodes in the graph
    mo_nodes = [(n, d) for n, d in G.nodes(data=True)
                if str(d.get("type", "")).startswith("STN_MO")]

    total = len(mo_nodes)
    true_nodes = [(n, d) for n, d in mo_nodes if not d.get("is_noisy")]
    noisy_nodes = total - len(true_nodes)

    output_lines = []
    line = f"\033[33m{label} nodes total={total} true={len(true_nodes)} noisy={noisy_nodes}\033[0m"
    print(line)
    output_lines.append(line)

    # Group by run/series
    keys = sorted({d.get(by, "NA") for _, d in mo_nodes})

    for key in keys:
        group_nodes = [(n, d) for n, d in mo_nodes if d.get(by, "NA") == key]
        group_true = [(n, d) for n, d in group_nodes if not d.get("is_noisy")]
        group_noisy = len(group_nodes) - len(group_true)

        gens = sorted({int(d.get("gen_idx", -1)) for _, d in group_true})
        front_sizes = [int(d.get("front_size", 0)) for _, d in group_true]

        # Basic stats
        fs_sum = sum(front_sizes)
        fs_min = min(front_sizes) if front_sizes else 0
        fs_max = max(front_sizes) if front_sizes else 0
        fs_mean = (fs_sum / len(front_sizes)) if front_sizes else 0

        line = (
            f"\033[33m  {by} {key}: true={len(group_true)} noisy={group_noisy} "
            f"gens={len(gens)} | front_size sum={fs_sum}, mean={fs_mean:.2f}, "
            f"min={fs_min}, max={fs_max}\033[0m"
        )
        print(line)
        output_lines.append(line)

        if list_fronts and group_true:
            # List per-generation front sizes (first max_list entries)
            per_gen = sorted(
                (int(d.get("gen_idx", -1)), int(d.get("front_size", 0)))
                for _, d in group_true
            )
            shown = per_gen[:max_list]
            tail = " ..." if len(per_gen) > max_list else ""
            gen_str = ", ".join(f"G{g}:{sz}" for g, sz in shown)
            line = f"\033[33m    fronts (G: size): {gen_str}{tail}\033[0m"
            print(line)
            output_lines.append(line)

    return "\n".join(output_lines)

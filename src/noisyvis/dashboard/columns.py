"""
Column configuration for dashboard DataFrames.

This module centralizes all column name definitions used for:
- Hiding columns in table displays
- Selecting columns for specific views
- Dropping columns during transformations
- Deduplication keys
"""

# Result-row columns the dashboard does not read yet (MO recording plan v6, Work Group 5): the persistent
# multi-objective record is consumed through results.mo_view.MORunView once the consumers migrate
# (Work Group 6). Until then DashboardData.load drops it, so no table, store or callback carries it.
DEFERRED_MO_COLUMNS = [
    'mo_record',
]

# Columns to hide in LON table display
LON_HIDDEN_COLUMNS = [
    'problem_name',
    'problem_type',
    'problem_goal',
    'dimensions',
    'opt_global',
    'local_optima',
    'fitness_values',
    'edges',
    'optima_feasibility',
    'neighbour_feasibility',
    'visit_counts',
    'visit_proportions',
]

# Columns to hide in algorithm table (table 2)
DISPLAY2_HIDDEN_COLUMNS = [
    'problem_type',
    'problem_goal',
    'problem_name',
    'dimensions',
    'opt_global',
    'PID',
    'experiment_name',
    'experiment_description',
]

# Columns to show in problem selection table (table 1)
DISPLAY1_COLUMNS = [
    'problem_type',
    'problem_goal',
    'problem_name',
    'dimensions',
    'opt_global',
    'fit_func',
    'PID',
]

# Columns containing list data (excluded from df_no_lists)
# These columns contain per-run trajectory/solution data
LIST_COLUMNS = [
    'rep_sols',
    'rep_fits',
    'rep_noisy_sols',
    'rep_fitness_boxplot_stats',
    'rep_noisy_fits',
    'rep_estimated_fits_whenadopted',
    'rep_estimated_fits_whendiscarded',
    'count_estimated_fits_whenadopted',
    'count_estimated_fits_whendiscarded',
    'sol_iterations',
    'sol_iterations_evals',
    'sol_transitions',
    'alternative_rep_sols',
    'alternative_rep_fits',
    'pareto_solutions',
    'pareto_fitnesses',
    'pareto_true_fitnesses',
    'true_pareto_solutions',
    'true_pareto_fitnesses',
    'noisy_pf_noisy_hypervolumes',
    'noisy_pf_true_hypervolumes',
    'true_pf_hypervolumes',
    'n_gens_pareto_best',
    'mo_record',  # also dropped at load (DEFERRED_MO_COLUMNS); never a table column
]

# Columns to drop from display2_df (algorithm selection table)
# These are either list columns or run-specific data not needed for selection
DISPLAY2_DROP_COLUMNS = [
    'n_gens',
    'n_evals',
    'stop_trigger',
    'n_unique_sols',
    'rep_sols',
    'rep_fits',
    'rep_noisy_sols',
    'rep_fitness_boxplot_stats',
    'rep_noisy_fits',
    'rep_estimated_fits_whenadopted',
    'rep_estimated_fits_whendiscarded',
    'count_estimated_fits_whenadopted',
    'count_estimated_fits_whendiscarded',
    'final_fit',
    'max_fit',
    'min_fit',
    'sol_iterations',
    'sol_iterations_evals',
    'sol_transitions',
    'alternative_rep_sols',
    'alternative_rep_fits',
    'seed',
    'seed_signature',
    'pareto_solutions',
    'pareto_fitnesses',
    'pareto_true_fitnesses',
    'true_pareto_solutions',
    'true_pareto_fitnesses',
    'noisy_pf_noisy_hypervolumes',
    'noisy_pf_true_hypervolumes',
    'true_pf_hypervolumes',
    'n_gens_pareto_best',
    'final_true_hv',
    'max_true_hv',
    'min_true_hv',
    'final_noisy_pf_hv',
    'max_noisy_pf_hv',
    'min_noisy_pf_hv',
    # MO recording plan v6, Work Group 5: per-run final-generation summaries stored beside mo_record
    'final_hv_current_noisy_front__noisy',
    'final_hv_current_noisy_front__clean',
    'final_hv_current_clean_front__clean',
    'final_hv_noisy_archive__noisy',
    'final_hv_noisy_archive__clean',
    'final_hv_clean_archive__clean',
    'n_generated_genotypes',
    'final_noisy_archive_size',
    'final_clean_archive_size',
    'mo_record',
    'run_id',
    'parent_run_id',
    'payload_path',
    'peak_ram_mb',
    'penalty',
]

# Keys used for deduplication in display2_df
# These columns together identify a unique algorithm configuration
DISPLAY2_DEDUP_KEYS = [
    'PID',
    'algo_type',
    'algo_name',
    'noise',
    'fit_func',
]

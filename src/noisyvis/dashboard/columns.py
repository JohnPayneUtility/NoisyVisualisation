"""
Column configuration for dashboard DataFrames.

This module centralizes all column name definitions used for:
- Hiding columns in table displays
- Selecting columns for specific views
- Dropping columns during transformations
- Deduplication keys
"""

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
    # the multi-objective record stays on the server (dashboard.mo_frames); never a table column
    'mo_record',
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
    # per-run final-generation summaries of the multi-objective record (MO recording plan v6)
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

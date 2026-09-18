"""The dashboard and MLflow-app contracts, characterised before the Stage 11 split (plan 11-0).

Stage 11 splits `dashboard/Dashboard.py` into `dashboard/app.py` plus `dashboard/callbacks/*`,
replaces the wildcard imports, resolves the duplicate callback names (B9), splits
`dataio/transformers.py` into `analysis/misjudgements.py` and `dashboard/tables.py`, moves
`column_config.py` to `dashboard/columns.py`, dissolves `dataio/` into `dashboard/data.py`, and
renames `noisyvis/app/` to `noisyvis/mlflow_app/` with `MLRUNS_DIR` in place of `parents[4]`.

None of that may change what the application does. The Stage-10 suite pins the visualisation
pipeline; the reproducibility baselines never execute any of this code. These tests pin what
neither can see:

 1. the Dash callback contract: 41 callbacks, their registration order, every Output/Input/State,
    `prevent_initial_call`, and the digest the live `/_dash-dependencies` endpoint serves;
 2. each callback's body by normalised AST and string-literal multiset, hashed under a fixed name so
    a B9 rename does not move the hash, with the name and the defining module pinned separately;
 3. every module global each callback reads, resolved to the object it names, so replacing the six
    wildcard imports with explicit ones can be shown to bind the same objects;
 4. every top-level statement of the seven files Stage 11 redistributes, each found exactly once,
    at its PRE or its planned POST location (`tests/harness/dashboard_inventory.py`);
 5. the app configuration, the layout digest and its 110 component ids, and that
    `DashboardData.load()` still runs exactly once per application import (R9);
 6. every callback driven over the real Dash HTTP protocol, plus a chained run of the populated
    pipeline: selection -> STN/LON stores -> the one shared-graph orchestrator (I-10c);
 7. the DataFrame schemas of the four table builders and the column constants;
 8. the import surface: the loaded third-party packages and every existing explicit import, which
    amendment A1 preserves, so Stage 11 has no import-time delta;
 9. the MLflow browser app: its page registry, dependencies, layout and `MLRUNS_DIR` equivalence;
10. the entrypoints: the Compose commands, the `[project.scripts]` targets, and the rule that no
    module may import an entrypoint module (under `-m` it is `__main__`, so an import would load a
    second copy of the app and of the 1.3 GB warehouse).

The EXPECTED values are frozen literals captured from a `git archive` of PRE_STAGE_11 (fa8d250) in
fresh subprocesses under the runner's Python 3.11, twice, with identical results. They were never
derived from a tree that Stage 11 had touched. That commit passes the Stage-10 `BODIES` contract, so
these values are anchored to PRE_STAGE_10 as well.

The probe writes a small synthetic warehouse into the harness temp root and lets the **real**
`DashboardData.load()` read it through `NOISYVIS_ROOT`: nothing is monkeypatched, and the production
warehouse is never opened. The knapsack PID is a real instance, so the LON half of the pipeline runs
for real. `PYTHONHASHSEED` is pinned to 0 for the probe subprocess only, as in test_viz_package.

Locations were accepted at their PRE or their intended POST module while `ALLOW_PRE_LOCATIONS` was
True, exactly as the Stage-10 harness did. Checkpoint 11-I tightened the dashboard side to POST only;
the MLflow app keeps its own `ALLOW_PRE_MLFLOW_LOCATIONS` until 11-J.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness import dashboard_inventory as inv
from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root

PRE_STAGE_11_COMMIT = "fa8d250c65fb9b778a18262f165967476f824f98"

# The dashboard side (Dashboard.py, DashboardHelpers.py, dataio/): final locations only from 11-I.
ALLOW_PRE_LOCATIONS = False
# The MLflow browser app (app/ -> mlflow_app/): checkpoint 11-J sets this to False.
ALLOW_PRE_MLFLOW_LOCATIONS = True

PKG = WORKSPACE / "src" / "noisyvis"
INVENTORY_PATH = HARNESS_DIR / "dashboard_inventory.py"

# --------------------------------------------------------------- frozen at PRE_STAGE_11 (fa8d250)

CALLBACK_ORDER = ['..schematic-graph.figure...schematic-legend.children..',
 'experiment-description-display.children',
 'knapsack-info-display.children',
 'table1.data',
 'table1-selected-store.data',
 'table2-selected-store.data',
 'data-problem-specific.data',
 'table2.data',
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..',
 'table2-selected-output.children',
 'penalty-filter-dropdown.options',
 'penalty-filter-dropdown.value',
 'plot_2d_data.data',
 'hide-series-dropdown.options',
 '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..',
 'plot_2d_data_table.data',
 '2DLinePlot.figure',
 '2DBoxPlot.figure',
 '2DLinePlotMO.figure',
 '2DBoxPlotMO.figure',
 '2DLinePlotEvalsSO.figure',
 '2DBoxPlotEvalsSO.figure',
 '2DBoxPlotPenaltySO.figure',
 '2DBoxPlotMisjudgementsSO.figure',
 '2DBoxAdvancedMisjudgementsSO.figure',
 'misjudgements-summary-table.children',
 'performance-summary-table.children',
 'mann-whitney-table.children',
 'evals-summary-table.children',
 'evals-mann-whitney-table.children',
 '2DPlotTabContent.children',
 'lon-table-selected-pid-store.data',
 'LON_data.data',
 'STN_data.data',
 '..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..',
 'run-print-info.style',
 'print_STN_series_labels.children',
 'axis-values.data',
 '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..',
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..',
 'plotParetoFront.figure']

CALLBACK_LIST_DIGEST = 'b7a4a16319eb00fd832e7bb6a7a7ab64b605e8ad80954c4142453a2f83d1dd39'

DEPENDENCIES_DIGEST = 'd50097e2d22e719cb510eb8bd7e9c84c7db3e12a5116fa16059e4862d3e7c953'

CALLBACKS = {'..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': {'ast': '1793c27aedfc8e3d2fd69cef326b81ad0b86ab0f46759ef63f2c025a0ee98b4c',
                                                                                                                                                        'inputs': ['STN_data.data',
                                                                                                                                                                   'mo_plot_type.value'],
                                                                                                                                                        'literals': '837e8d5b88f3efde50184fe8ed40b24368a05a93b22c553d8e31f6579c3b5021',
                                                                                                                                                        'module': 'dashboard/Dashboard.py',
                                                                                                                                                        'name': 'process_STN_data',
                                                                                                                                                        'order': 34,
                                                                                                                                                        'params': ['df',
                                                                                                                                                                   'mo_plot_type',
                                                                                                                                                                   'group_cols'],
                                                                                                                                                        'prevent_initial_call': False,
                                                                                                                                                        'state': []},
 '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..': {'ast': '36c7684358ce5e5590b28e5d48051f0f611a2bc8391980e0edb12cc010ca52e6',
                                                                                                 'inputs': ['plot_2d_data.data'],
                                                                                                 'literals': '24fd5cbee18b0697ae6bc6bc33cbbf437090948e4cd663a7a8600487271f6345',
                                                                                                 'module': 'dashboard/Dashboard.py',
                                                                                                 'name': 'update_advanced_misjudgement_algo_options',
                                                                                                 'order': 14,
                                                                                                 'params': ['data'],
                                                                                                 'prevent_initial_call': False,
                                                                                                 'state': []},
 '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..': {'ast': '4157d323196921e525f190eeca5136dd9d82bdb381b5910425b939a14094c8af',
                                                                                        'inputs': ['annotation-options.value'],
                                                                                        'literals': 'be02d83664a644a710d01376409d09088dda76cc66c428247c7429a1c6a67743',
                                                                                        'module': 'dashboard/Dashboard.py',
                                                                                        'name': 'handle_print_mode',
                                                                                        'order': 38,
                                                                                        'params': ['annotation_options'],
                                                                                        'prevent_initial_call': True,
                                                                                        'state': []},
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': {'ast': 'bc02fc217291ae2667d86c361468d2c978b2cbf951082608d9f173d977a46f1c',
                                                                       'inputs': ['data-problem-specific.data',
                                                                                  'lon-table-selected-pid-store.data'],
                                                                       'literals': 'c591691b3c665d373600daf21c35260a75ad73f670f3a882490bc9d20b21b7d5',
                                                                       'module': 'dashboard/Dashboard.py',
                                                                       'name': 'update_table2',
                                                                       'order': 8,
                                                                       'params': ['data',
                                                                                  'lon_table_pid',
                                                                                  'table1_selection'],
                                                                       'prevent_initial_call': False,
                                                                       'state': ['table1-selected-store.data']},
 '..schematic-graph.figure...schematic-legend.children..': {'ast': '076c374aa62db7534060aa779ec9fe1412f844f300e413e94594fb1bf1a6bf08',
                                                            'inputs': ['schematic-misjudgements.value',
                                                                       'schematic-simple-annotations.value',
                                                                       'schematic-boxplots.value'],
                                                            'literals': '0e3b014f049ba53f64ea43d264b48f84027ce8619d275bd329e8a811aca34f44',
                                                            'module': 'dashboard/Dashboard.py',
                                                            'name': 'update_schematic',
                                                            'order': 0,
                                                            'params': ['misjudgement_values',
                                                                       'simple_values',
                                                                       'boxplot_values'],
                                                            'prevent_initial_call': False,
                                                            'state': []},
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': {'ast': '9d7e90e48a252a6fb5d50dbc2eadcecf011f599e190248182477a8628926c733',
                                                                                                                                                                                                                           'inputs': ['optimum.data',
                                                                                                                                                                                                                                      'PID.data',
                                                                                                                                                                                                                                      'opt_goal.data',
                                                                                                                                                                                                                                      'options.value',
                                                                                                                                                                                                                                      'run-options.value',
                                                                                                                                                                                                                                      'STN_lower_fit_limit.value',
                                                                                                                                                                                                                                      'LON-fit-percent.value',
                                                                                                                                                                                                                                      'LON-options.value',
                                                                                                                                                                                                                                      'LON-node-colour-mode.value',
                                                                                                                                                                                                                                      'LON-surface-colour.value',
                                                                                                                                                                                                                                      'LON-edge-colour-feas.value',
                                                                                                                                                                                                                                      'lmds-multiplier.value',
                                                                                                                                                                                                                                      'NLON_fit_func.value',
                                                                                                                                                                                                                                      'NLON_intensity.value',
                                                                                                                                                                                                                                      'NLON_samples.value',
                                                                                                                                                                                                                                      'NLON_penalty.value',
                                                                                                                                                                                                                                      'layout.value',
                                                                                                                                                                                                                                      'plotType.value',
                                                                                                                                                                                                                                      'hover-info.value',
                                                                                                                                                                                                                                      'azimuth_deg.value',
                                                                                                                                                                                                                                      'elevation_deg.value',
                                                                                                                                                                                                                                      'STN_data_processed.data',
                                                                                                                                                                                                                                      'STN_series_labels.data',
                                                                                                                                                                                                                                      'run-index.value',
                                                                                                                                                                                                                                      'run-selector.value',
                                                                                                                                                                                                                                      'LON_data.data',
                                                                                                                                                                                                                                      'axis-values.data',
                                                                                                                                                                                                                                      'opacity_noise_bar.value',
                                                                                                                                                                                                                                      'LON_node_opacity.value',
                                                                                                                                                                                                                                      'LON_edge_opacity.value',
                                                                                                                                                                                                                                      'STN_node_opacity.value',
                                                                                                                                                                                                                                      'STN_edge_opacity.value',
                                                                                                                                                                                                                                      'STN-node-min.value',
                                                                                                                                                                                                                                      'STN-node-max.value',
                                                                                                                                                                                                                                      'LON-node-min.value',
                                                                                                                                                                                                                                      'LON-node-max.value',
                                                                                                                                                                                                                                      'LON-edge-size-slider.value',
                                                                                                                                                                                                                                      'STN-edge-size-slider.value',
                                                                                                                                                                                                                                      'noisy_fitnesses_data.data',
                                                                                                                                                                                                                                      'stn-plot-type.value',
                                                                                                                                                                                                                                      'STN_MO_data.data',
                                                                                                                                                                                                                                      'STN_MO_series_labels.data',
                                                                                                                                                                                                                                      'stn-node-size-metric.value',
                                                                                                                                                                                                                                      'annotation-options.value',
                                                                                                                                                                                                                                      'fit_func_store.data',
                                                                                                                                                                                                                                      'info-panel-x.value',
                                                                                                                                                                                                                                      'info-panel-y.value',
                                                                                                                                                                                                                                      'axes-text-scale.value',
                                                                                                                                                                                                                                      'annotation-text-scale.value',
                                                                                                                                                                                                                                      'plot-theme.value',
                                                                                                                                                                                                                                      'plot_2d_data.data',
                                                                                                                                                                                                                                      'lon-scatter-x-axis.value',
                                                                                                                                                                                                                                      'lon-scatter-y-axis.value',
                                                                                                                                                                                                                                      'lon-scatter-plot-style.value',
                                                                                                                                                                                                                                      'lon-scatter-multi-noise.value'],
                                                                                                                                                                                                                           'literals': 'f51280651ead6cd2aabe12d5dfd46f47adc2e919410cbc538c757a78bf83328c',
                                                                                                                                                                                                                           'module': 'dashboard/Dashboard.py',
                                                                                                                                                                                                                           'name': 'update_plot',
                                                                                                                                                                                                                           'order': 39,
                                                                                                                                                                                                                           'params': ['optimum',
                                                                                                                                                                                                                                      'PID',
                                                                                                                                                                                                                                      'opt_goal',
                                                                                                                                                                                                                                      'options',
                                                                                                                                                                                                                                      'run_options',
                                                                                                                                                                                                                                      'STN_lower_fit_limit',
                                                                                                                                                                                                                                      'LO_fit_percent',
                                                                                                                                                                                                                                      'LON_options',
                                                                                                                                                                                                                                      'LON_node_colour_mode',
                                                                                                                                                                                                                                      'LON_surface_colour',
                                                                                                                                                                                                                                      'LON_edge_colour_feas',
                                                                                                                                                                                                                                      'lmds_multiplier',
                                                                                                                                                                                                                                      'NLON_fit_func',
                                                                                                                                                                                                                                      'NLON_intensity',
                                                                                                                                                                                                                                      'NLON_samples',
                                                                                                                                                                                                                                      'NLON_penalty',
                                                                                                                                                                                                                                      'layout_value',
                                                                                                                                                                                                                                      'plot_type',
                                                                                                                                                                                                                                      'hover_info_value',
                                                                                                                                                                                                                                      'azimuth_deg',
                                                                                                                                                                                                                                      'elevation_deg',
                                                                                                                                                                                                                                      'all_trajectories_list',
                                                                                                                                                                                                                                      'STN_labels',
                                                                                                                                                                                                                                      'run_start_index',
                                                                                                                                                                                                                                      'n_runs_display',
                                                                                                                                                                                                                                      'local_optima',
                                                                                                                                                                                                                                      'axis_values',
                                                                                                                                                                                                                                      'opacity_noise_bar',
                                                                                                                                                                                                                                      'LON_node_opacity',
                                                                                                                                                                                                                                      'LON_edge_opacity',
                                                                                                                                                                                                                                      'STN_node_opacity',
                                                                                                                                                                                                                                      'STN_edge_opacity',
                                                                                                                                                                                                                                      'STN_node_min',
                                                                                                                                                                                                                                      'STN_node_max',
                                                                                                                                                                                                                                      'LON_node_min',
                                                                                                                                                                                                                                      'LON_node_max',
                                                                                                                                                                                                                                      'LON_edge_size_slider',
                                                                                                                                                                                                                                      'STN_edge_size_slider',
                                                                                                                                                                                                                                      'noisy_fitnesses_list',
                                                                                                                                                                                                                                      'stn_plot_type',
                                                                                                                                                                                                                                      'STN_MO_data',
                                                                                                                                                                                                                                      'STN_MO_series_labels',
                                                                                                                                                                                                                                      'stn_node_size_metric',
                                                                                                                                                                                                                                      'annotation_options',
                                                                                                                                                                                                                                      'fit_func',
                                                                                                                                                                                                                                      'info_panel_x',
                                                                                                                                                                                                                                      'info_panel_y',
                                                                                                                                                                                                                                      'axes_text_scale',
                                                                                                                                                                                                                                      'annotation_text_scale',
                                                                                                                                                                                                                                      'plot_theme',
                                                                                                                                                                                                                                      'plot_2d_data',
                                                                                                                                                                                                                                      'lon_scatter_x',
                                                                                                                                                                                                                                      'lon_scatter_y',
                                                                                                                                                                                                                                      'lon_scatter_plot_style',
                                                                                                                                                                                                                                      'lon_scatter_multi_noise'],
                                                                                                                                                                                                                           'prevent_initial_call': False,
                                                                                                                                                                                                                           'state': []},
 '2DBoxAdvancedMisjudgementsSO.figure': {'ast': 'c53d16ee5a0a4774f2af690a8318cf4813881213a63e7fbd9901b771556705f1',
                                         'inputs': ['plot_2d_data.data',
                                                    'fit_func_store.data',
                                                    'plot-theme.value',
                                                    'noise-cap-input.value',
                                                    'advanced-misjudgement-algo-dropdown.value'],
                                         'literals': 'dcc02d643887c936b6c06e68da40b6c4ee11afee66cc5751693a0340971b76e6',
                                         'module': 'dashboard/Dashboard.py',
                                         'name': 'display_box_advanced_misjudgements_so',
                                         'order': 24,
                                         'params': ['data',
                                                    'fit_func',
                                                    'plot_theme',
                                                    'noise_cap',
                                                    'selected_algo'],
                                         'prevent_initial_call': False,
                                         'state': []},
 '2DBoxPlot.figure': {'ast': 'acb27c8f3ec18578d072adfdd46328f20fb542928279fbe05aba797335842f09',
                      'inputs': ['plot_2d_data.data',
                                 'so-fitness-mode.value',
                                 'fit_func_store.data',
                                 'opt_goal.data',
                                 'plot-theme.value',
                                 'noise-cap-input.value',
                                 'hide-series-dropdown.value'],
                      'literals': 'f801d78d14043150868f9f281477905b807206eaa445af647ce38a7623477fff',
                      'module': 'dashboard/Dashboard.py',
                      'name': 'display_stored_data',
                      'order': 17,
                      'params': ['data',
                                 'fitness_mode',
                                 'fit_func',
                                 'opt_goal',
                                 'plot_theme',
                                 'noise_cap',
                                 'hidden_series'],
                      'prevent_initial_call': False,
                      'state': []},
 '2DBoxPlotEvalsSO.figure': {'ast': '18fdba4c3eb2d5846ebbd2450449a1fe6e2a1e3715c9bb0d5f35678f11f38ea6',
                             'inputs': ['plot_2d_data.data',
                                        'so-fitness-mode.value',
                                        'fit_func_store.data',
                                        'plot-theme.value',
                                        'noise-cap-input.value',
                                        'hide-series-dropdown.value'],
                             'literals': '23745611122f4e23cfeccc28566a7383e377d773885690c1b52514d71d923f24',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_box_evals_so',
                             'order': 21,
                             'params': ['data',
                                        'fitness_mode',
                                        'fit_func',
                                        'plot_theme',
                                        'noise_cap',
                                        'hidden_series'],
                             'prevent_initial_call': False,
                             'state': []},
 '2DBoxPlotMO.figure': {'ast': '6fb95bf039790ceb1f2ecd28bfb5dac1e4456ae419cc920810eb77e1311e9f15',
                        'inputs': ['plot_2d_data.data',
                                   'plot-theme.value',
                                   'noise-cap-input.value',
                                   'hide-series-dropdown.value'],
                        'literals': '5cec1a92bc632029cb9a8437918566fd9818cf7b3eee5734838c588346ab38d2',
                        'module': 'dashboard/Dashboard.py',
                        'name': 'display_stored_data_mo_box',
                        'order': 19,
                        'params': ['data', 'plot_theme', 'noise_cap', 'hidden_series'],
                        'prevent_initial_call': False,
                        'state': []},
 '2DBoxPlotMisjudgementsSO.figure': {'ast': '59efa97308d1d097903d825ce83739af4f8dc9bc402203527e8a1d0a4efa76dc',
                                     'inputs': ['plot_2d_data.data',
                                                'fit_func_store.data',
                                                'plot-theme.value',
                                                'noise-cap-input.value',
                                                'hide-series-dropdown.value'],
                                     'literals': 'efe1d13479cbc103bfbcfd55ecba4dde3abd878628100a47ab32202070e62428',
                                     'module': 'dashboard/Dashboard.py',
                                     'name': 'display_box_misjudgements_so',
                                     'order': 23,
                                     'params': ['data',
                                                'fit_func',
                                                'plot_theme',
                                                'noise_cap',
                                                'hidden_series'],
                                     'prevent_initial_call': False,
                                     'state': []},
 '2DBoxPlotPenaltySO.figure': {'ast': '4bc732571cd8502fc347fd707ab4b60f78bfdf4ec8515b068f09bc3f3a953176',
                               'inputs': ['plot_2d_data.data',
                                          'so-fitness-mode.value',
                                          'fit_func_store.data',
                                          'opt_goal.data',
                                          'plot-theme.value',
                                          'noise-cap-input.value',
                                          'hide-series-dropdown.value'],
                               'literals': 'a7032ad9cdb29ff56de57f86366e79fdbaf06be366c903e83e3cd3a4e71ab988',
                               'module': 'dashboard/Dashboard.py',
                               'name': 'display_box_penalty_so',
                               'order': 22,
                               'params': ['data',
                                          'fitness_mode',
                                          'fit_func',
                                          'opt_goal',
                                          'plot_theme',
                                          'noise_cap',
                                          'hidden_series'],
                               'prevent_initial_call': False,
                               'state': []},
 '2DLinePlot.figure': {'ast': 'b2acd659bc64434aa8fe73cb72347ec74877d3d5d9655c2ba43d8733b1e180ce',
                       'inputs': ['plot_2d_data.data',
                                  'so-fitness-mode.value',
                                  'fit_func_store.data',
                                  'opt_goal.data',
                                  'plot-theme.value',
                                  'noise-cap-input.value',
                                  'hide-series-dropdown.value'],
                       'literals': '19431fe5f2fa249e07b1c70b9f0b56ffa550e5eee4b702e47d9f466c3b893177',
                       'module': 'dashboard/Dashboard.py',
                       'name': 'display_stored_data',
                       'order': 16,
                       'params': ['data',
                                  'fitness_mode',
                                  'fit_func',
                                  'opt_goal',
                                  'plot_theme',
                                  'noise_cap',
                                  'hidden_series'],
                       'prevent_initial_call': False,
                       'state': []},
 '2DLinePlotEvalsSO.figure': {'ast': '82f0f41d78ced2e79023e783c79a96af3459c22bab711fe643d33f2e6eaf60bf',
                              'inputs': ['plot_2d_data.data',
                                         'line-evals-show-std.value',
                                         'so-fitness-mode.value',
                                         'fit_func_store.data',
                                         'plot-theme.value',
                                         'noise-cap-input.value',
                                         'hide-series-dropdown.value'],
                              'literals': 'c9c904491d1ea9cb678d313eb3a583dd8b23876489efb0d2a2178e02efd04f7a',
                              'module': 'dashboard/Dashboard.py',
                              'name': 'display_line_evals_so',
                              'order': 20,
                              'params': ['data',
                                         'std_checkbox',
                                         'fitness_mode',
                                         'fit_func',
                                         'plot_theme',
                                         'noise_cap',
                                         'hidden_series'],
                              'prevent_initial_call': False,
                              'state': []},
 '2DLinePlotMO.figure': {'ast': 'f282c64355b427096f931e9eb4cdb4b278ea1765c7f4030bd266c9003a5373bd',
                         'inputs': ['plot_2d_data.data',
                                    'plot-theme.value',
                                    'noise-cap-input.value',
                                    'hide-series-dropdown.value'],
                         'literals': '6f2b0462509663d0533b5a84a8895f3d02f7704bb617a879105735be71563232',
                         'module': 'dashboard/Dashboard.py',
                         'name': 'display_stored_data_mo_line',
                         'order': 18,
                         'params': ['data', 'plot_theme', 'noise_cap', 'hidden_series'],
                         'prevent_initial_call': False,
                         'state': []},
 '2DPlotTabContent.children': {'ast': '7c997ecb910b42dc7693a87e3d75fa2f00d047061eb91d4aca709f3169a0d3f5',
                               'inputs': ['2DPlotTabSelection.value'],
                               'literals': '7e996af29e6dd93a6cdca04c45a0736a41440cd6a8c6802428972d1a11784571',
                               'module': 'dashboard/Dashboard.py',
                               'name': 'render_content_2DPlot_tab',
                               'order': 30,
                               'params': ['tab'],
                               'prevent_initial_call': False,
                               'state': []},
 'LON_data.data': {'ast': '83f6408221d6460dee9767b4223812c5d6da1787938c5e6a30f3928f675dd11c',
                   'inputs': ['LON_table.selected_rows'],
                   'literals': '6fb69bf7487dff26ecab22eff8d3804674436f1899944081d67b102b26d8d469',
                   'module': 'dashboard/Dashboard.py',
                   'name': 'update_filtered_view',
                   'order': 32,
                   'params': ['selected_rows', 'LON_table_data'],
                   'prevent_initial_call': False,
                   'state': ['LON_table.data']},
 'STN_data.data': {'ast': '6702ad38db765439edb27c9de4f58911dfe6f112a38958cf79d1660ab156139c',
                   'inputs': ['table2.selected_rows', 'penalty-filter-dropdown.value'],
                   'literals': 'a4e1ecfec49831194e6d4f8cc40d23a063262a6a0e5cb6f9a9565103b5a7cea5',
                   'module': 'dashboard/Dashboard.py',
                   'name': 'update_filtered_view',
                   'order': 33,
                   'params': ['selected_rows', 'penalty_value', 'table2_data'],
                   'prevent_initial_call': False,
                   'state': ['table2.data']},
 'axis-values.data': {'ast': '7fe03f80678e3990382aa0d41c3d44cfb43ec5fdcaefd562455ccaca2b5a648b',
                      'inputs': ['custom_x_min.value',
                                 'custom_x_max.value',
                                 'custom_y_min.value',
                                 'custom_y_max.value',
                                 'custom_z_min.value',
                                 'custom_z_max.value',
                                 'log-z-axis.value'],
                      'literals': 'a89565368bc3c07d609e2e3c23f34e6ea7418f08816f87b0a4abbaf161010609',
                      'module': 'dashboard/Dashboard.py',
                      'name': 'clean_axis_values',
                      'order': 37,
                      'params': ['custom_x_min',
                                 'custom_x_max',
                                 'custom_y_min',
                                 'custom_y_max',
                                 'custom_z_min',
                                 'custom_z_max',
                                 'log_z'],
                      'prevent_initial_call': False,
                      'state': []},
 'data-problem-specific.data': {'ast': '61a03bf97b260a5642d9b068abc99d63886a8816c5467cd9a85b1a94f5e6c8e9',
                                'inputs': ['table1-selected-store.data',
                                           'experiment-selector.value'],
                                'literals': 'c3b38d3553e756dc3bc1cc061e0a0673af81722dc847d56086a8ed28105e24e0',
                                'module': 'dashboard/Dashboard.py',
                                'name': 'filter_table2',
                                'order': 6,
                                'params': ['selection1', 'experiment_name', 'table1_current_data'],
                                'prevent_initial_call': False,
                                'state': ['table1.data']},
 'evals-mann-whitney-table.children': {'ast': 'b8d39e74208759c81b1b82e987b3b27b2f41cc476205b93c942457a4e2837a08',
                                       'inputs': ['plot_2d_data.data',
                                                  'fit_func_store.data',
                                                  'so-fitness-mode.value'],
                                       'literals': '77854eb3232e7d3808ceb6a4dd1c6cf72635e95a3d1cea30b0bfe3d4383cfecb',
                                       'module': 'dashboard/Dashboard.py',
                                       'name': 'update_evals_mann_whitney_table',
                                       'order': 29,
                                       'params': ['data', 'fit_func', 'fitness_mode'],
                                       'prevent_initial_call': False,
                                       'state': []},
 'evals-summary-table.children': {'ast': 'd6f972121864a7fc3ff8c9035f39f18455acf426e6fa8568e0f9b3f712f610c8',
                                  'inputs': ['plot_2d_data.data',
                                             'fit_func_store.data',
                                             'so-fitness-mode.value',
                                             'round-stats-checkbox.value',
                                             'scientific-notation-checkbox.value'],
                                  'literals': '77d3ce03f655a9260ed6ccbb24a99020011fad0fc164044c729d372865b44e59',
                                  'module': 'dashboard/Dashboard.py',
                                  'name': 'update_evals_summary_table',
                                  'order': 28,
                                  'params': ['data',
                                             'fit_func',
                                             'fitness_mode',
                                             'round_stats_value',
                                             'sci_value'],
                                  'prevent_initial_call': False,
                                  'state': []},
 'experiment-description-display.children': {'ast': '0de74c69ded5634c6379815f297e7189afec63fa92b78972ccb638b99a3e65c4',
                                             'inputs': ['experiment-selector.value'],
                                             'literals': '350110652918a2abef4376bbff03d6518232fc48a5bb471792f49ff0e95332ed',
                                             'module': 'dashboard/Dashboard.py',
                                             'name': 'update_experiment_description',
                                             'order': 1,
                                             'params': ['selected'],
                                             'prevent_initial_call': False,
                                             'state': []},
 'hide-series-dropdown.options': {'ast': 'a576b35b1843c7e2d49c1c50a4ab20bd925732e8a862e769ceeebb8e6a273f9a',
                                  'inputs': ['plot_2d_data.data'],
                                  'literals': '380ade669b2031677644ac25cc03bfd506566f3ad866081fee5a9fdcf8a49555',
                                  'module': 'dashboard/Dashboard.py',
                                  'name': 'update_hide_series_options',
                                  'order': 13,
                                  'params': ['data'],
                                  'prevent_initial_call': False,
                                  'state': []},
 'knapsack-info-display.children': {'ast': 'db122f62e170454ab4264a3dfda6b64722b73a7fc0f01bf4339a99bc82fae20c',
                                    'inputs': ['PID.data'],
                                    'literals': 'b522a25146ececf5a5a524b3a39d96cc4c7ca788be198138d4764a96ca4b519a',
                                    'module': 'dashboard/Dashboard.py',
                                    'name': 'update_knapsack_info',
                                    'order': 2,
                                    'params': ['pid'],
                                    'prevent_initial_call': False,
                                    'state': []},
 'lon-table-selected-pid-store.data': {'ast': 'a2073fb51dd8918de38fd515baf0d56d97102513a279e4c4a57cb13ba6271c91',
                                       'inputs': ['LON_table.selected_rows'],
                                       'literals': '01a7ee181a3b25ad2a3586c9dc94a01055d01aa1b9f83770c18df477cf8ce1b9',
                                       'module': 'dashboard/Dashboard.py',
                                       'name': 'update_lon_table_selected_pid',
                                       'order': 31,
                                       'params': ['selected_rows', 'lon_table_data'],
                                       'prevent_initial_call': True,
                                       'state': ['LON_table.data']},
 'mann-whitney-table.children': {'ast': 'e817dda83bb0f13f50a9d65c4556e8b8dd67ee34f4044cecfbc67ecdc8d11979',
                                 'inputs': ['plot_2d_data.data',
                                            'so-fitness-mode.value',
                                            'fit_func_store.data',
                                            'opt_goal.data'],
                                 'literals': '3380bfd4df668613711496b082798d07727450a36508102f2344475f213767a3',
                                 'module': 'dashboard/Dashboard.py',
                                 'name': 'update_mann_whitney_table',
                                 'order': 27,
                                 'params': ['data', 'fitness_mode', 'fit_func', 'opt_goal'],
                                 'prevent_initial_call': False,
                                 'state': []},
 'misjudgements-summary-table.children': {'ast': '1b0f22b3bdcf9b9ecac685499f061aaff8a6967be21e670508cd1d651ad3f711',
                                          'inputs': ['plot_2d_data.data',
                                                     'fit_func_store.data',
                                                     'round-stats-checkbox.value',
                                                     'scientific-notation-checkbox.value'],
                                          'literals': 'c59734fd17bb04d7057c60a314a7ca8b84612ca4565b8d376c089686ca3b8a86',
                                          'module': 'dashboard/Dashboard.py',
                                          'name': 'update_misjudgements_summary_table',
                                          'order': 25,
                                          'params': ['data',
                                                     'fit_func',
                                                     'round_stats_value',
                                                     'sci_value'],
                                          'prevent_initial_call': False,
                                          'state': []},
 'penalty-filter-dropdown.options': {'ast': '1fc2ac755429e8bd246cc5bb84fe155cf4c25723a4ff5466ad3361f795be1296',
                                     'inputs': ['table2.derived_virtual_data'],
                                     'literals': '99d97ede88ca6c0a103be0a4935a21ad36316767b9b309b2a435da78edca924e',
                                     'module': 'dashboard/Dashboard.py',
                                     'name': 'update_penalty_filter_options',
                                     'order': 10,
                                     'params': ['filtered_data'],
                                     'prevent_initial_call': False,
                                     'state': []},
 'penalty-filter-dropdown.value': {'ast': '982b66f93a8ac9fe66eb5a20940534a6add2a234d46eff563bf16f6c11a1c1c6',
                                   'inputs': ['PID.data'],
                                   'literals': 'a628237ddfcf55403967b83a24f0e64309508617587e0e091dbf8a0e540510fd',
                                   'module': 'dashboard/Dashboard.py',
                                   'name': 'reset_penalty_filter_on_problem_change',
                                   'order': 11,
                                   'params': ['_pid'],
                                   'prevent_initial_call': True,
                                   'state': []},
 'performance-summary-table.children': {'ast': 'f881a9fe71e5d2f7f9773ef872b3b01f4f8ddabf3a3df9edf209157298d51d7a',
                                        'inputs': ['plot_2d_data.data',
                                                   'so-fitness-mode.value',
                                                   'fit_func_store.data',
                                                   'opt_goal.data',
                                                   'round-stats-checkbox.value',
                                                   'scientific-notation-checkbox.value'],
                                        'literals': 'f632c4402670a99df36cd880f239ae59e6e7a776438007a3471a8f802f165acf',
                                        'module': 'dashboard/Dashboard.py',
                                        'name': 'update_performance_summary_table',
                                        'order': 26,
                                        'params': ['data',
                                                   'fitness_mode',
                                                   'fit_func',
                                                   'opt_goal',
                                                   'round_stats_value',
                                                   'sci_value'],
                                        'prevent_initial_call': False,
                                        'state': []},
 'plotParetoFront.figure': {'ast': '103b38c81aa3dd1e532cba4ca74702f3ec13a9025392cae17df34b32fca856fd',
                            'inputs': ['MO_data_PPP.data',
                                       'STN_MO_series_labels.data',
                                       'paretoFrontPlotType.value',
                                       'IndVsDist_IndType.value',
                                       'IndVsDist_DistType.value',
                                       'paretoPlotNumRuns.value',
                                       'paretoPlotWindowSize.value'],
                            'literals': '3de33cb4cedf04ac528635238127681a8f5ef39ed76a880f88949a73462a3429',
                            'module': 'dashboard/Dashboard.py',
                            'name': 'updateParetoPlot',
                            'order': 40,
                            'params': ['frontdata',
                                       'series_labels',
                                       'paretoFrontPlotType',
                                       'IndVsDist_IndType',
                                       'IndVsDist_DistType',
                                       'nruns',
                                       'windowSize'],
                            'prevent_initial_call': False,
                            'state': []},
 'plot_2d_data.data': {'ast': 'ad7f46051502e6864be95a64817a6a4646cdcb1514afcfeb25e7ae1897fc1516',
                       'inputs': ['table2.derived_virtual_data', 'penalty-filter-dropdown.value'],
                       'literals': 'f5a8360cf0142d8e2f58bebc2957497cbc67159c1e9eedf9770dd8d95ade57a5',
                       'module': 'dashboard/Dashboard.py',
                       'name': 'update_filtered_view',
                       'order': 12,
                       'params': ['filtered_data', 'penalty_value'],
                       'prevent_initial_call': False,
                       'state': []},
 'plot_2d_data_table.data': {'ast': '3fd8ea62a334feebc8a41d49d27408b12187c4e08e81c023f38254762406ddd5',
                             'inputs': ['plot_2d_data.data'],
                             'literals': '99e07f820cca60d1ff9ffa2781aa29296a1c955fbde32c6d2898d28758e0468a',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data',
                             'order': 15,
                             'params': ['data'],
                             'prevent_initial_call': False,
                             'state': []},
 'print_STN_series_labels.children': {'ast': 'eeb9a7a6ff294aac7e3428c0584d1f133527a65957ebf34f74d4307117e1efba',
                                      'inputs': ['STN_series_labels.data'],
                                      'literals': '48a4b42e43cd07e2cbe8613e36cb3f150da005b7424a5fd8f4be09f36a333c27',
                                      'module': 'dashboard/Dashboard.py',
                                      'name': 'update_table2_selected',
                                      'order': 36,
                                      'params': ['series_list'],
                                      'prevent_initial_call': False,
                                      'state': []},
 'run-print-info.style': {'ast': 'a124c407d64598f7fdad775e2ac59a1955cd69e808b139d2a26fb4565a262744',
                          'inputs': ['show-text-info.value'],
                          'literals': '6bfa310fd1a4e7fff91cd0235d7c0db86b65e9165127b0054bd52eb70131e66b',
                          'module': 'dashboard/Dashboard.py',
                          'name': 'toggle_run_print_info',
                          'order': 35,
                          'params': ['value'],
                          'prevent_initial_call': False,
                          'state': []},
 'table1-selected-store.data': {'ast': '70c3b4967270375bde9a11035d207b926dbcc9dd96693c2d57795d5f2b45487b',
                                'inputs': ['table1.selected_rows', 'experiment-selector.value'],
                                'literals': '8c0e77a8694491358038ff6baad885125e2ddaca687b62f57d54d436f725bc5d',
                                'module': 'dashboard/Dashboard.py',
                                'name': 'update_table1_store',
                                'order': 4,
                                'params': ['selected_rows', '_experiment_name'],
                                'prevent_initial_call': True,
                                'state': []},
 'table1.data': {'ast': '81cef4e5a989fee7f826d4e9bcd81b6f204915348d95c026ae6b8aa30c84a39f',
                 'inputs': ['experiment-selector.value'],
                 'literals': '23ae3dab8b9a447fdf9d2ebf061acaf83af505251f520efce1c7dc6d5a0942b1',
                 'module': 'dashboard/Dashboard.py',
                 'name': 'update_table1_for_experiment',
                 'order': 3,
                 'params': ['experiment_name'],
                 'prevent_initial_call': False,
                 'state': []},
 'table2-selected-output.children': {'ast': '603bba5ef1ab85795fe31aa99075a088b5abdb6ee59bd2ef28761000d62e124d',
                                     'inputs': ['table2.selected_rows'],
                                     'literals': '6ef6f2390d25f3b8a5d5e7bfc7a33dd972352adc4f553a5f31e1c91a71dc0c72',
                                     'module': 'dashboard/Dashboard.py',
                                     'name': 'update_table2_selected',
                                     'order': 9,
                                     'params': ['selected_rows', 'table2_data'],
                                     'prevent_initial_call': False,
                                     'state': ['table2.data']},
 'table2-selected-store.data': {'ast': 'cea4bd4ff95cdde649ca04575cfd3da85d6050e706933972a30370f73d43bf99',
                                'inputs': ['table2.selected_rows'],
                                'literals': 'd0eae5a51258a857c7a6d7c28857b223f76a9b3e0383a618893c0477784f94fa',
                                'module': 'dashboard/Dashboard.py',
                                'name': 'update_table2_store',
                                'order': 5,
                                'params': ['selected_rows'],
                                'prevent_initial_call': True,
                                'state': []},
 'table2.data': {'ast': '1a5a51de2b2857f85054e19aeb58d2fe2ed49428c4acdcf91db9f835cbe363f6',
                 'inputs': ['data-problem-specific.data'],
                 'literals': 'd251a0cd16f9c8229a11f14d3d43e01740b8b9169e74277da1f02b0a09c6e9a0',
                 'module': 'dashboard/Dashboard.py',
                 'name': 'update_table2',
                 'order': 7,
                 'params': ['data'],
                 'prevent_initial_call': False,
                 'state': []}}

B9_DUPLICATES = {'display_stored_data': ['2DBoxPlot.figure', '2DLinePlot.figure', 'plot_2d_data_table.data'],
 'update_filtered_view': ['LON_data.data', 'STN_data.data', 'plot_2d_data.data'],
 'update_table2': ['..optimum.data...PID.data...opt_goal.data...fit_func_store.data..',
                   'table2.data'],
 'update_table2_selected': ['print_STN_series_labels.children', 'table2-selected-output.children']}

CONFIG = {'assets_folder': 'src/noisyvis/dashboard/assets',
 'assets_folder_exists': False,
 'assets_url_path': 'assets',
 'include_assets_files': True,
 'prevent_initial_callbacks': False,
 'serve_locally': True,
 'suppress_callback_exceptions': True,
 'title': 'Dash',
 'update_title': 'Updating...',
 'url_base_pathname': None,
 'use_pages': False}

LAYOUT_DIGEST = '1adc784f96e8ae5cfb116eda16262145897fdb016a401a25d845d7b734db78a6'

LAYOUT_IDS = ['2DPlotTabContent',
 '2DPlotTabSelection',
 'IndVsDist_DistType',
 'IndVsDist_IndType',
 'LON-edge-colour-feas',
 'LON-edge-size-slider',
 'LON-fit-percent',
 'LON-node-colour-mode',
 'LON-node-max',
 'LON-node-min',
 'LON-options',
 'LON-surface-colour',
 'LON_data',
 'LON_edge_opacity',
 'LON_node_opacity',
 'LON_table',
 'MO_data_PPP',
 'NLON_fit_func',
 'NLON_intensity',
 'NLON_penalty',
 'NLON_samples',
 'PID',
 'STN-edge-size-slider',
 'STN-node-max',
 'STN-node-min',
 'STN_MO_data',
 'STN_MO_series_labels',
 'STN_data',
 'STN_data_processed',
 'STN_edge_opacity',
 'STN_lower_fit_limit',
 'STN_node_opacity',
 'STN_series_labels',
 'advanced-misjudgement-algo-dropdown',
 'annotation-options',
 'annotation-text-scale',
 'axes-text-scale',
 'axis-values',
 'azimuth_deg',
 'custom_x_max',
 'custom_x_min',
 'custom_y_max',
 'custom_y_min',
 'custom_z_max',
 'custom_z_min',
 'data-problem-specific',
 'elevation_deg',
 'evals-mann-whitney-table',
 'evals-summary-table',
 'experiment-description-display',
 'experiment-selector',
 'fit_func_store',
 'hide-series-dropdown',
 'hover-info',
 'info-panel-x',
 'info-panel-y',
 'knapsack-info-display',
 'layout',
 'lmds-multiplier',
 'log-z-axis',
 'lon-feas-error-correlations',
 'lon-feas-error-scatter',
 'lon-scatter-multi-noise',
 'lon-scatter-plot-style',
 'lon-scatter-x-axis',
 'lon-scatter-y-axis',
 'lon-selected-correlation',
 'lon-stats-table',
 'lon-table-selected-pid-store',
 'mann-whitney-table',
 'misjudgements-summary-table',
 'mo_plot_type',
 'noise-cap-input',
 'noisy_fitnesses_data',
 'opacity_noise_bar',
 'opt_goal',
 'optimum',
 'options',
 'paretoFrontPlotType',
 'paretoPlotNumRuns',
 'paretoPlotWindowSize',
 'penalty-filter-dropdown',
 'performance-summary-table',
 'plot-theme',
 'plotParetoFront',
 'plotType',
 'plot_2d_data',
 'print_STN_series_labels',
 'round-stats-checkbox',
 'run-index',
 'run-options',
 'run-print-info',
 'run-selector',
 'schematic-boxplots',
 'schematic-graph',
 'schematic-legend',
 'schematic-misjudgements',
 'schematic-simple-annotations',
 'scientific-notation-checkbox',
 'show-text-info',
 'so-fitness-mode',
 'stn-node-size-metric',
 'stn-plot-type',
 'stn-stats-table',
 'table1',
 'table1-selected-store',
 'table2',
 'table2-selected-output',
 'table2-selected-store',
 'trajectory-plot']

LOAD_COUNT = {'algo': 1, 'lon': 1}

GLOBALS = {'bindings': {'noisyvis.dashboard.Dashboard:LON_display_columns': 'lon_display_columns',
              'noisyvis.dashboard.Dashboard:df': 'df',
              'noisyvis.dashboard.Dashboard:df_LONs': 'df_lon',
              'noisyvis.dashboard.Dashboard:df_no_lists': 'df_no_lists',
              'noisyvis.dashboard.Dashboard:display1_df': 'display1_df',
              'noisyvis.dashboard.Dashboard:display2_df': 'display2_df'},
 'data_module': 'noisyvis.dataio',
 'holder': 'noisyvis.dashboard.Dashboard'}

SCRIPTS = {'noisyvis-dashboard': 'ModuleNotFoundError', 'noisyvis-mlflow': 'ModuleNotFoundError'}

FREE_NAMES = {'..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': {'pd': {'kind': 'module',
                                                                                                                                                               'name': 'pandas'}},
 '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..': {'pd': {'kind': 'module',
                                                                                                        'name': 'pandas'}},
 '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..': {'dash': {'kind': 'module',
                                                                                                 'name': 'dash'}},
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': {'LON_TABLE_SELECTED_PID_STORE': {'canon': 'lon-table-selected-pid-store',
                                                                                                        'kind': 'value'},
                                                                       'pd': {'kind': 'module',
                                                                              'name': 'pandas'}},
 '..schematic-graph.figure...schematic-legend.children..': {'_build_schematic_figure': {'ast': '15a2fa923415041815674d30119c17ab4c7033228e5bc0c01e092e3dda53b9e6',
                                                                                        'kind': 'callable',
                                                                                        'name': '_build_schematic_figure'},
                                                            '_build_schematic_legend': {'ast': '513e5c5d45b66c64432fbf71917d354787c60b3850df4cc0bbb82304d2cf0977',
                                                                                        'kind': 'callable',
                                                                                        'name': '_build_schematic_legend'}},
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': {'LON_SCATTER_AXIS_LABELS': {'canon': {'dict': [['neigh_feas',
                                                                                                                                                                                                                                                                           'Neighbourhood '
                                                                                                                                                                                                                                                                           'Feasibility'],
                                                                                                                                                                                                                                                                          ['error',
                                                                                                                                                                                                                                                                           'Sampling '
                                                                                                                                                                                                                                                                           'Error'],
                                                                                                                                                                                                                                                                          ['abs_error',
                                                                                                                                                                                                                                                                           'Absolute '
                                                                                                                                                                                                                                                                           'Error'],
                                                                                                                                                                                                                                                                          ['iqr',
                                                                                                                                                                                                                                                                           'Sample '
                                                                                                                                                                                                                                                                           'Range '
                                                                                                                                                                                                                                                                           '(Q3-Q1)'],
                                                                                                                                                                                                                                                                          ['fitness',
                                                                                                                                                                                                                                                                           'Fitness'],
                                                                                                                                                                                                                                                                          ['median',
                                                                                                                                                                                                                                                                           'Median '
                                                                                                                                                                                                                                                                           'Sampled '
                                                                                                                                                                                                                                                                           'Fitness']]},
                                                                                                                                                                                                                                                       'kind': 'value'},
                                                                                                                                                                                                                           'LON_SCATTER_DEFAULT_PLOT_STYLE': {'canon': 'scatter',
                                                                                                                                                                                                                                                              'kind': 'value'},
                                                                                                                                                                                                                           'LON_SCATTER_DEFAULT_X_AXIS': {'canon': 'neigh_feas',
                                                                                                                                                                                                                                                          'kind': 'value'},
                                                                                                                                                                                                                           'LON_SCATTER_DEFAULT_Y_AXIS': {'canon': 'error',
                                                                                                                                                                                                                                                          'kind': 'value'},
                                                                                                                                                                                                                           '_add_guide_nodes': {'ast': '5a4991aa3826cbc58cc3c04cd8a4a85b73cdd5cb224e561821f1ddf8e21c43f3',
                                                                                                                                                                                                                                                'kind': 'callable',
                                                                                                                                                                                                                                                'name': '_add_guide_nodes'},
                                                                                                                                                                                                                           '_build_stn_stats_table': {'ast': 'a7460f7aafb20e55505edc6a95dce0e4c6e116bd4c86f62cc63b92a5c4eb94f9',
                                                                                                                                                                                                                                                      'kind': 'callable',
                                                                                                                                                                                                                                                      'name': '_build_stn_stats_table'},
                                                                                                                                                                                                                           '_get_noise_param_label': {'ast': '8138f0ee5c7dec1bac04a364492ecaaea7efc9881617cd15c590074925088292',
                                                                                                                                                                                                                                                      'kind': 'callable',
                                                                                                                                                                                                                                                      'name': '_get_noise_param_label'},
                                                                                                                                                                                                                           'add_lon_edges': {'ast': '5c6f73c6f71997c13e9b1452614e546adb34cf291956e851d25b1963ddd040d1',
                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                             'name': 'add_lon_edges'},
                                                                                                                                                                                                                           'add_lon_nodes': {'ast': '16374f54f260ab1e2fee43f985834ce62309305b8cc8fbcd9fde6a709a5a1eb5',
                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                             'name': 'add_lon_nodes'},
                                                                                                                                                                                                                           'add_mo_fronts': {'ast': 'd7d09e7be956b3ab28c5c69c410b4d3c508d8916c6dd009e84730d0953f9d788',
                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                             'name': 'add_mo_fronts'},
                                                                                                                                                                                                                           'add_prior_noise_stn_algo_pov': {'ast': '62496cb4794186ccf7b1fbd987e708bd40e2fa8fd2bfab25f0871667c0befcf8',
                                                                                                                                                                                                                                                            'kind': 'callable',
                                                                                                                                                                                                                                                            'name': 'add_prior_noise_stn_algo_pov'},
                                                                                                                                                                                                                           'add_prior_noise_stn_v4': {'ast': 'c42e2f94f099b62623e1462a2b54f6fc956db9b4f5536412afac9556ec1486d6',
                                                                                                                                                                                                                                                      'kind': 'callable',
                                                                                                                                                                                                                                                      'name': 'add_prior_noise_stn_v4'},
                                                                                                                                                                                                                           'add_prior_noise_stn_v5': {'ast': '4fd049f24ea1e06140d2f54103162df757a9b10482244b7cfa3997729104a01a',
                                                                                                                                                                                                                                                      'kind': 'callable',
                                                                                                                                                                                                                                                      'name': 'add_prior_noise_stn_v5'},
                                                                                                                                                                                                                           'add_stn_trajectories': {'ast': '18818678708be3d6a31954bf43fadd7d453843d32a390c773116911fcd9c318b',
                                                                                                                                                                                                                                                    'kind': 'callable',
                                                                                                                                                                                                                                                    'name': 'add_stn_trajectories'},
                                                                                                                                                                                                                           'build_all_traces': {'ast': 'bbf7a780d64370f7afd256b5936c88108af496985a4fd6621068f61359bff168',
                                                                                                                                                                                                                                                'kind': 'callable',
                                                                                                                                                                                                                                                'name': 'build_all_traces'},
                                                                                                                                                                                                                           'build_correlation_table': {'ast': 'a470beefdd4ef2dba56e076be9b3c67db8df1b0f9071bac568d6a3fc657090f5',
                                                                                                                                                                                                                                                       'kind': 'callable',
                                                                                                                                                                                                                                                       'name': 'build_correlation_table'},
                                                                                                                                                                                                                           'build_selected_correlation_display': {'ast': '48fbfa39552765bff9d0c3d72e9c4019914853101e68add1ce530f40f2308cb3',
                                                                                                                                                                                                                                                                  'kind': 'callable',
                                                                                                                                                                                                                                                                  'name': 'build_selected_correlation_display'},
                                                                                                                                                                                                                           'calculate_lon_statistics': {'ast': '6560bdcd253d53dac15d3376894ba6c60d0049f0bed6b4c495ae69f7e4b04af0',
                                                                                                                                                                                                                                                        'kind': 'callable',
                                                                                                                                                                                                                                                        'name': 'calculate_lon_statistics'},
                                                                                                                                                                                                                           'calculate_positions': {'ast': '860cd1cb5ad743ff17302e125b20070fcae3363c6b31ca0d4090f42325a55de3',
                                                                                                                                                                                                                                                   'kind': 'callable',
                                                                                                                                                                                                                                                   'name': 'calculate_positions'},
                                                                                                                                                                                                                           'comparison_misjudgement_step_indices': {'ast': '49dbc10692f2e1c0d57768151637c1bd57072c058f4c41159379c745eff27a17',
                                                                                                                                                                                                                                                                    'kind': 'callable',
                                                                                                                                                                                                                                                                    'name': 'comparison_misjudgement_step_indices'},
                                                                                                                                                                                                                           'compute_correlation_pair': {'ast': 'f1f64f65296abc44db293a60823598354ac8709865b8049cf503e0297de26546',
                                                                                                                                                                                                                                                        'kind': 'callable',
                                                                                                                                                                                                                                                        'name': 'compute_correlation_pair'},
                                                                                                                                                                                                                           'compute_node_feasibility_error': {'ast': '33f16584712f6f4cc8657512d9f97c22e28c68dd7fc3eaa46af9efc24a22f7d2',
                                                                                                                                                                                                                                                              'kind': 'callable',
                                                                                                                                                                                                                                                              'name': 'compute_node_feasibility_error'},
                                                                                                                                                                                                                           'compute_pairwise_correlations': {'ast': '289a9f3d3dc61a22f85e907cea87064c94623b22acc3ad08cd4dc8dfd5657501',
                                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                                             'name': 'compute_pairwise_correlations'},
                                                                                                                                                                                                                           'constraint_misjudgement_step_indices': {'ast': '5079976cc5cb40cedbc53ef8fd53e5adf708c44f8ff676acbcb798ead6c49cae',
                                                                                                                                                                                                                                                                    'kind': 'callable',
                                                                                                                                                                                                                                                                    'name': 'constraint_misjudgement_step_indices'},
                                                                                                                                                                                                                           'convert_to_single_edges_format': {'ast': 'c2aa9bc982f76b6e0b4db751a90e892f1a37b4163ec5967970c293b75e5c081c',
                                                                                                                                                                                                                                                              'kind': 'callable',
                                                                                                                                                                                                                                                              'name': 'convert_to_single_edges_format'},
                                                                                                                                                                                                                           'create_axis_settings': {'ast': '7db8637c06d7e280bb577779ddab3db65160061a748bc419160fde3176eeea83',
                                                                                                                                                                                                                                                    'kind': 'callable',
                                                                                                                                                                                                                                                    'name': 'create_axis_settings'},
                                                                                                                                                                                                                           'create_figure': {'ast': 'b9f77986cfda392064269e7e1c1cbce066c4fc02dd904bb0ba5f037f5e46154b',
                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                             'name': 'create_figure'},
                                                                                                                                                                                                                           'create_guide_traces': {'ast': '4b601205206e5b59e9d278af57be55e15271acede44ecd99e61144fecf06b8b8',
                                                                                                                                                                                                                                                   'kind': 'callable',
                                                                                                                                                                                                                                                   'name': 'create_guide_traces'},
                                                                                                                                                                                                                           'dash_table': {'kind': 'module',
                                                                                                                                                                                                                                          'name': 'dash.dash_table'},
                                                                                                                                                                                                                           'dataclass_replace': {'ast': '5acb1e995024f6a66ba151df2a6bd81582edd0738a546abc53349a5f3a406bc7',
                                                                                                                                                                                                                                                 'kind': 'callable',
                                                                                                                                                                                                                                                 'name': 'replace'},
                                                                                                                                                                                                                           'debug_mo_counts': {'ast': 'c463074b4ea8f62d2554f140c1bcc73a9bc26978d1e09a864f69746a73ec976c',
                                                                                                                                                                                                                                               'kind': 'callable',
                                                                                                                                                                                                                                               'name': 'debug_mo_counts'},
                                                                                                                                                                                                                           'filter_local_optima': {'ast': 'c5ca48a389db20b6275e538dbfacb26d5d6675a1a51f6f2008fcd53e63cac1cc',
                                                                                                                                                                                                                                                   'kind': 'callable',
                                                                                                                                                                                                                                                   'name': 'filter_local_optima'},
                                                                                                                                                                                                                           'filter_negative_LO': {'ast': '4a78f9bfcc0e5f0a54fbf8c1288cbd3fb53d419058f8db6e8fc678049f81ee3c',
                                                                                                                                                                                                                                                  'kind': 'callable',
                                                                                                                                                                                                                                                  'name': 'filter_negative_LO'},
                                                                                                                                                                                                                           'generate_run_summary_string': {'ast': '6c7689af454a79cd0988ba32b2bd5c4fa524ef38ea318fe60edc6405de0b4f29',
                                                                                                                                                                                                                                                           'kind': 'callable',
                                                                                                                                                                                                                                                           'name': 'generate_run_summary_string'},
                                                                                                                                                                                                                           'get_mean_run': {'ast': 'd00157fbcb3ca236874aa02cd388d1107037bb8743cead7d101a231df654664c',
                                                                                                                                                                                                                                            'kind': 'callable',
                                                                                                                                                                                                                                            'name': 'get_mean_run'},
                                                                                                                                                                                                                           'get_median_run': {'ast': 'd402e0019a63781d6ef03f018ab25e84bb3cb7f108458f1274384dcabd8c3fce',
                                                                                                                                                                                                                                              'kind': 'callable',
                                                                                                                                                                                                                                              'name': 'get_median_run'},
                                                                                                                                                                                                                           'go': {'kind': 'module',
                                                                                                                                                                                                                                  'name': 'plotly.graph_objects'},
                                                                                                                                                                                                                           'html': {'kind': 'module',
                                                                                                                                                                                                                                    'name': 'dash.html'},
                                                                                                                                                                                                                           'increasing_noise_step_indices': {'ast': '26b6dbd0e4f38a23a642968a1877050b151fb02f5f9eb1a5b7f2351a7cb0dd55',
                                                                                                                                                                                                                                                             'kind': 'callable',
                                                                                                                                                                                                                                                             'name': 'increasing_noise_step_indices'},
                                                                                                                                                                                                                           'nx': {'kind': 'module',
                                                                                                                                                                                                                                  'name': 'networkx'},
                                                                                                                                                                                                                           'parse_callback_inputs': {'ast': 'b3211930c9b4e6d78f5c6ef5b39b6793911b31be5b3150fcb3f1e7e6d533d899',
                                                                                                                                                                                                                                                     'kind': 'callable',
                                                                                                                                                                                                                                                     'name': 'parse_callback_inputs'},
                                                                                                                                                                                                                           'pd': {'kind': 'module',
                                                                                                                                                                                                                                  'name': 'pandas'},
                                                                                                                                                                                                                           'plot_lon_stats': {'ast': '3cf45315991737c2e0bd23c2cd733afa000a14add3f4cff505279f49cb749dc8',
                                                                                                                                                                                                                                              'kind': 'callable',
                                                                                                                                                                                                                                              'name': 'plot_lon_stats'},
                                                                                                                                                                                                                           'plot_lon_stats_multi': {'ast': '272fbb0b7a5c90b3a7d2c47f04ce75d02f78cc7ff509abd6a942579048040815',
                                                                                                                                                                                                                                                    'kind': 'callable',
                                                                                                                                                                                                                                                    'name': 'plot_lon_stats_multi'},
                                                                                                                                                                                                                           'px': {'kind': 'module',
                                                                                                                                                                                                                                  'name': 'plotly.express'},
                                                                                                                                                                                                                           'select_top_runs_by_fitness': {'ast': 'f7551569ea5c14a0048252e0a19c1db48486616f53d09b202ea62b8ebf984470',
                                                                                                                                                                                                                                                          'kind': 'callable',
                                                                                                                                                                                                                                                          'name': 'select_top_runs_by_fitness'},
                                                                                                                                                                                                                           'style_nodes': {'ast': 'c5c2d640448f6ab7dd351810746c9c323f54cd8182a301433bf482903152f02b',
                                                                                                                                                                                                                                           'kind': 'callable',
                                                                                                                                                                                                                                           'name': 'style_nodes'}},
 '2DBoxAdvancedMisjudgementsSO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                                        'kind': 'callable',
                                                        'name': '_cap_noise'},
                                         '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                                                 'kind': 'callable',
                                                                 'name': '_get_so_xaxis_label'},
                                         'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                                         'pd': {'kind': 'module', 'name': 'pandas'},
                                         'plot2d_box_advanced_misjudgements_so': {'ast': 'ee1f1585217b35d9553d4ef450ebde063fef616af2f551a8e3704d82298744ea',
                                                                                  'kind': 'callable',
                                                                                  'name': 'plot_box_advanced_misjudgements_so'}},
 '2DBoxPlot.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                     'kind': 'callable',
                                     'name': '_cap_noise'},
                      '_get_problem_goal': {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                                            'kind': 'callable',
                                            'name': '_get_problem_goal'},
                      '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                              'kind': 'callable',
                                              'name': '_get_so_xaxis_label'},
                      '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                       'kind': 'callable',
                                       'name': '_hide_series'},
                      'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                      'pd': {'kind': 'module', 'name': 'pandas'},
                      'plot2d_box': {'ast': '404cad1d7cb32fd7003e9806b1b5306b55107e61c6d91d2d216b0c6403ebab83',
                                     'kind': 'callable',
                                     'name': 'plot_box'}},
 '2DBoxPlotEvalsSO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                            'kind': 'callable',
                                            'name': '_cap_noise'},
                             '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                                     'kind': 'callable',
                                                     'name': '_get_so_xaxis_label'},
                             '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                              'kind': 'callable',
                                              'name': '_hide_series'},
                             'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                             'pd': {'kind': 'module', 'name': 'pandas'},
                             'plot2d_box_evals': {'ast': '7f72d7aadff43076dde88caf850035a44a4c705e413d47e7dc6e816e1aa2ca64',
                                                  'kind': 'callable',
                                                  'name': 'plot_box_evals'}},
 '2DBoxPlotMO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                       'kind': 'callable',
                                       'name': '_cap_noise'},
                        '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                         'kind': 'callable',
                                         'name': '_hide_series'},
                        'pd': {'kind': 'module', 'name': 'pandas'},
                        'plot2d_box_mo': {'ast': '0455279f7a4c88b5b0bd3b95293ad16b9f4d321473706414d8f00720f7779dee',
                                          'kind': 'callable',
                                          'name': 'plot_box_mo'}},
 '2DBoxPlotMisjudgementsSO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                                    'kind': 'callable',
                                                    'name': '_cap_noise'},
                                     '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                                             'kind': 'callable',
                                                             'name': '_get_so_xaxis_label'},
                                     '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                                      'kind': 'callable',
                                                      'name': '_hide_series'},
                                     'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                                     'pd': {'kind': 'module', 'name': 'pandas'},
                                     'plot2d_box_misjudgements_so': {'ast': 'fc101959c79d3912dd47c02ab8acddf501817187ae6c4f3c686f19d5adb9c454',
                                                                     'kind': 'callable',
                                                                     'name': 'plot_box_misjudgements_so'}},
 '2DBoxPlotPenaltySO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                              'kind': 'callable',
                                              'name': '_cap_noise'},
                               '_get_problem_goal': {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                                                     'kind': 'callable',
                                                     'name': '_get_problem_goal'},
                               '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                                'kind': 'callable',
                                                'name': '_hide_series'},
                               'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                               'pd': {'kind': 'module', 'name': 'pandas'},
                               'plot2d_box_penalty': {'ast': 'bf3d58160ec51aa00f33134129d525d0d6a967886430b31f8964547e95834357',
                                                      'kind': 'callable',
                                                      'name': 'plot_box_penalty'}},
 '2DLinePlot.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                      'kind': 'callable',
                                      'name': '_cap_noise'},
                       '_get_problem_goal': {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                                             'kind': 'callable',
                                             'name': '_get_problem_goal'},
                       '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                               'kind': 'callable',
                                               'name': '_get_so_xaxis_label'},
                       '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                        'kind': 'callable',
                                        'name': '_hide_series'},
                       'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                       'pd': {'kind': 'module', 'name': 'pandas'},
                       'plot2d_line': {'ast': 'cafd6966fcb48402fa63c19ab77c6a0c2ac9979e4ec3541a306cfd6493c68ef0',
                                       'kind': 'callable',
                                       'name': 'plot_line'}},
 '2DLinePlotEvalsSO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                             'kind': 'callable',
                                             'name': '_cap_noise'},
                              '_get_so_xaxis_label': {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                                                      'kind': 'callable',
                                                      'name': '_get_so_xaxis_label'},
                              '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                               'kind': 'callable',
                                               'name': '_hide_series'},
                              'go': {'kind': 'module', 'name': 'plotly.graph_objects'},
                              'pd': {'kind': 'module', 'name': 'pandas'},
                              'plot2d_line_evals': {'ast': 'eb0485a5ef0edbe43c3361dd5af2a3a71d9cb517217be904fe6f349791ff4158',
                                                    'kind': 'callable',
                                                    'name': 'plot_line_evals'}},
 '2DLinePlotMO.figure': {'_cap_noise': {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                                        'kind': 'callable',
                                        'name': '_cap_noise'},
                         '_hide_series': {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                                          'kind': 'callable',
                                          'name': '_hide_series'},
                         'pd': {'kind': 'module', 'name': 'pandas'},
                         'plot2d_line_mo': {'ast': 'd244a841dbe682f342bb92294ac1a234096e16992270a1b6f86f16554a1c0632',
                                            'kind': 'callable',
                                            'name': 'plot_line_mo'}},
 '2DPlotTabContent.children': {'dash_table': {'kind': 'module', 'name': 'dash.dash_table'},
                               'dcc': {'kind': 'module', 'name': 'dash.dcc'},
                               'df_no_lists': {'attr': 'df_no_lists', 'kind': 'data'},
                               'html': {'kind': 'module', 'name': 'dash.html'}},
 'LON_data.data': {'LON_display_columns': {'attr': 'lon_display_columns', 'kind': 'data'},
                   'convert_to_split_edges_format': {'ast': 'cf3ae769a9afc9602014c11d5555f521b4906933f2adfe0d9494137fd69ca9c9',
                                                     'kind': 'callable',
                                                     'name': 'convert_to_split_edges_format'},
                   'df_LONs': {'attr': 'df_lon', 'kind': 'data'},
                   'pd': {'kind': 'module', 'name': 'pandas'}},
 'STN_data.data': {'_filter_penalty': {'ast': '61a29fac7aec0852cafc06c3945077f876c9d83e9627898b2b15ee0b0a4b1beb',
                                       'kind': 'callable',
                                       'name': '_filter_penalty'},
                   'df': {'attr': 'df', 'kind': 'data'},
                   'pd': {'kind': 'module', 'name': 'pandas'}},
 'axis-values.data': {},
 'data-problem-specific.data': {'_filter_by_experiment': {'ast': '9d434d28e3dfd80dbfd760764b502f98d54060785855ada5d1a8b71e12db9642',
                                                          'kind': 'callable',
                                                          'name': '_filter_by_experiment'},
                                'create_display2_df': {'ast': 'e6df04a70682f8837a0ec4936dda966ff5013c78d0fe99df9fa8efdffb993be1',
                                                       'kind': 'callable',
                                                       'name': 'create_display2_df'},
                                'df': {'attr': 'df', 'kind': 'data'}},
 'evals-mann-whitney-table.children': {'_resolve_evals_column': {'ast': '15bdd0ba1415dcdbe74d1ec19063fd97b51b83460a9aebf4026a5d5d405f7b56',
                                                                 'kind': 'callable',
                                                                 'name': '_resolve_evals_column'},
                                       'dash_table': {'kind': 'module', 'name': 'dash.dash_table'},
                                       'dcc': {'kind': 'module', 'name': 'dash.dcc'},
                                       'html': {'kind': 'module', 'name': 'dash.html'},
                                       'pd': {'kind': 'module', 'name': 'pandas'},
                                       'tab_selected_style': {'canon': {'dict': [['height', '30px'],
                                                                                 ['lineHeight',
                                                                                  '30px'],
                                                                                 ['fontSize',
                                                                                  '14px'],
                                                                                 ['padding', '0px'],
                                                                                 ['backgroundColor',
                                                                                  '#ddd']]},
                                                              'kind': 'value'},
                                       'tab_style': {'canon': {'dict': [['height', '30px'],
                                                                        ['lineHeight', '30px'],
                                                                        ['fontSize', '14px'],
                                                                        ['padding', '0px']]},
                                                     'kind': 'value'}},
 'evals-summary-table.children': {'_format_median_std': {'ast': 'e7c35cd2186c7adcee6b7b8baa4aa97dc2133408c901e489521aa07c2860bbd3',
                                                         'kind': 'callable',
                                                         'name': '_format_median_std'},
                                  '_resolve_evals_column': {'ast': '15bdd0ba1415dcdbe74d1ec19063fd97b51b83460a9aebf4026a5d5d405f7b56',
                                                            'kind': 'callable',
                                                            'name': '_resolve_evals_column'},
                                  'dash_table': {'kind': 'module', 'name': 'dash.dash_table'},
                                  'html': {'kind': 'module', 'name': 'dash.html'},
                                  'pd': {'kind': 'module', 'name': 'pandas'}},
 'experiment-description-display.children': {'experiment_descriptions': {'canon': {'dict': [['exp-a',
                                                                                             'synthetic '
                                                                                             'knapsack']]},
                                                                         'kind': 'value'},
                                             'html': {'kind': 'module', 'name': 'dash.html'}},
 'hide-series-dropdown.options': {'pd': {'kind': 'module', 'name': 'pandas'}},
 'knapsack-info-display.children': {'get_knapsack_problem_stats': {'ast': 'c86a678013ee00e7bd28c4d5a71802815994c0ca03ad1b80d4e7f0d6e84b0bbd',
                                                                   'kind': 'callable',
                                                                   'name': 'get_knapsack_problem_stats'},
                                    'html': {'kind': 'module', 'name': 'dash.html'},
                                    'interpret_correlation': {'ast': '6d2c93e1d2d2614c477e13010f7ee8e2e62efec0ce802b14b0acfbbfd4ba7c0d',
                                                              'kind': 'callable',
                                                              'name': 'interpret_correlation'}},
 'lon-table-selected-pid-store.data': {'LON_TABLE_SELECTED_PID_STORE': {'canon': 'lon-table-selected-pid-store',
                                                                        'kind': 'value'}},
 'mann-whitney-table.children': {'_get_problem_goal': {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                                                       'kind': 'callable',
                                                       'name': '_get_problem_goal'},
                                 '_resolve_fit_column': {'ast': '0e0821be9500a2c1e1c87aea3025c154dc68d21fb9511f9b62223268aa63a02f',
                                                         'kind': 'callable',
                                                         'name': '_resolve_fit_column'},
                                 'dash_table': {'kind': 'module', 'name': 'dash.dash_table'},
                                 'dcc': {'kind': 'module', 'name': 'dash.dcc'},
                                 'html': {'kind': 'module', 'name': 'dash.html'},
                                 'pd': {'kind': 'module', 'name': 'pandas'},
                                 'tab_selected_style': {'canon': {'dict': [['height', '30px'],
                                                                           ['lineHeight', '30px'],
                                                                           ['fontSize', '14px'],
                                                                           ['padding', '0px'],
                                                                           ['backgroundColor',
                                                                            '#ddd']]},
                                                        'kind': 'value'},
                                 'tab_style': {'canon': {'dict': [['height', '30px'],
                                                                  ['lineHeight', '30px'],
                                                                  ['fontSize', '14px'],
                                                                  ['padding', '0px']]},
                                               'kind': 'value'}},
 'misjudgements-summary-table.children': {'_format_median_std': {'ast': 'e7c35cd2186c7adcee6b7b8baa4aa97dc2133408c901e489521aa07c2860bbd3',
                                                                 'kind': 'callable',
                                                                 'name': '_format_median_std'},
                                          'dash_table': {'kind': 'module',
                                                         'name': 'dash.dash_table'},
                                          'html': {'kind': 'module', 'name': 'dash.html'},
                                          'pd': {'kind': 'module', 'name': 'pandas'}},
 'penalty-filter-dropdown.options': {'_filter_by_table2_selection': {'ast': '23b8ec5ec7423f27a5e7e2e9b6515e934feed0eea26b8f7e88407d9b42816d78',
                                                                     'kind': 'callable',
                                                                     'name': '_filter_by_table2_selection'}},
 'penalty-filter-dropdown.value': {},
 'performance-summary-table.children': {'_format_median_std': {'ast': 'e7c35cd2186c7adcee6b7b8baa4aa97dc2133408c901e489521aa07c2860bbd3',
                                                               'kind': 'callable',
                                                               'name': '_format_median_std'},
                                        '_get_problem_goal': {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                                                              'kind': 'callable',
                                                              'name': '_get_problem_goal'},
                                        '_resolve_fit_column': {'ast': '0e0821be9500a2c1e1c87aea3025c154dc68d21fb9511f9b62223268aa63a02f',
                                                                'kind': 'callable',
                                                                'name': '_resolve_fit_column'},
                                        'dash_table': {'kind': 'module', 'name': 'dash.dash_table'},
                                        'html': {'kind': 'module', 'name': 'dash.html'},
                                        'pd': {'kind': 'module', 'name': 'pandas'}},
 'plotParetoFront.figure': {'get_pareto_plot': {'ast': '8a660705f72a757524625733cf214685af2dbe09d7c2bf1d58825c804ef4b1fc',
                                                'kind': 'callable',
                                                'name': 'get_pareto_plot'},
                            'go': {'kind': 'module', 'name': 'plotly.graph_objects'}},
 'plot_2d_data.data': {'_filter_by_table2_selection': {'ast': '23b8ec5ec7423f27a5e7e2e9b6515e934feed0eea26b8f7e88407d9b42816d78',
                                                       'kind': 'callable',
                                                       'name': '_filter_by_table2_selection'},
                       '_filter_penalty': {'ast': '61a29fac7aec0852cafc06c3945077f876c9d83e9627898b2b15ee0b0a4b1beb',
                                           'kind': 'callable',
                                           'name': '_filter_penalty'}},
 'plot_2d_data_table.data': {},
 'print_STN_series_labels.children': {},
 'run-print-info.style': {},
 'table1-selected-store.data': {'ctx': {'kind': 'object',
                                        'type': 'dash._callback_context.CallbackContext'}},
 'table1.data': {'DISPLAY1_COLUMNS': {'canon': {'list': ['problem_type',
                                                         'problem_goal',
                                                         'problem_name',
                                                         'dimensions',
                                                         'opt_global',
                                                         'fit_func',
                                                         'PID']},
                                      'kind': 'value'},
                 '_filter_by_experiment': {'ast': '9d434d28e3dfd80dbfd760764b502f98d54060785855ada5d1a8b71e12db9642',
                                           'kind': 'callable',
                                           'name': '_filter_by_experiment'},
                 'df': {'attr': 'df', 'kind': 'data'}},
 'table2-selected-output.children': {},
 'table2-selected-store.data': {},
 'table2.data': {'pd': {'kind': 'module', 'name': 'pandas'}}}

HTTP = {'..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': {'digest': 'a568383257859c07be83fb749dd368320e3f82c5753d738051e0f30c00ddc04d',
                                                                                                                                                        'length': 2303,
                                                                                                                                                        'props': ['MO_data_PPP.data',
                                                                                                                                                                  'STN_MO_data.data',
                                                                                                                                                                  'STN_MO_series_labels.data',
                                                                                                                                                                  'STN_data_processed.data',
                                                                                                                                                                  'STN_series_labels.data',
                                                                                                                                                                  'noisy_fitnesses_data.data'],
                                                                                                                                                        'status': 200,
                                                                                                                                                        'traces': {}},
 '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..': {'digest': 'edbdee8571013412dedd005f57baccd2bf3a29782a61db11cb8db34f786d8ff3',
                                                                                                 'length': 228,
                                                                                                 'props': ['advanced-misjudgement-algo-dropdown.options',
                                                                                                           'advanced-misjudgement-algo-dropdown.value'],
                                                                                                 'status': 200,
                                                                                                 'traces': {}},
 '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..': {'digest': '37ac9126bbf6956b9550a4d197d96a86c6f5297d08eb5a03f688669db9896503',
                                                                                        'length': 104,
                                                                                        'props': ['annotation-options.value'],
                                                                                        'status': 200,
                                                                                        'traces': {}},
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': {'digest': 'e8020fbd3d15bc0869ed50644fa2516952bb9f11cbfa0d550f4fab54cfbdbc4c',
                                                                       'length': 161,
                                                                       'props': ['PID.data',
                                                                                 'fit_func_store.data',
                                                                                 'opt_goal.data',
                                                                                 'optimum.data'],
                                                                       'status': 200,
                                                                       'traces': {}},
 '..schematic-graph.figure...schematic-legend.children..': {'digest': '43d6af601ea980b1896ba39536f59a6eb74bfe981333ef45ad81f7dd459d4731',
                                                            'length': 18036,
                                                            'props': ['schematic-graph.figure',
                                                                      'schematic-legend.children'],
                                                            'status': 200,
                                                            'traces': {'schematic-graph.figure': 27}},
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': {'digest': '304b729b67c2db797db35e407e93a45cd539069e33baa56dbd0cc46d9682213f',
                                                                                                                                                                                                                           'length': 15769,
                                                                                                                                                                                                                           'props': ['lon-feas-error-correlations.children',
                                                                                                                                                                                                                                     'lon-feas-error-scatter.figure',
                                                                                                                                                                                                                                     'lon-selected-correlation.children',
                                                                                                                                                                                                                                     'lon-stats-table.children',
                                                                                                                                                                                                                                     'run-print-info.children',
                                                                                                                                                                                                                                     'stn-stats-table.children',
                                                                                                                                                                                                                                     'trajectory-plot.figure'],
                                                                                                                                                                                                                           'status': 200,
                                                                                                                                                                                                                           'traces': {'lon-feas-error-scatter.figure': 0,
                                                                                                                                                                                                                                      'trajectory-plot.figure': 3}},
 '2DBoxAdvancedMisjudgementsSO.figure': {'digest': '16f4da18247a4e2e9d653ff3246793512649c8b324191d47d221255eda1220bf',
                                         'length': 8978,
                                         'props': ['2DBoxAdvancedMisjudgementsSO.figure'],
                                         'status': 200,
                                         'traces': {'2DBoxAdvancedMisjudgementsSO.figure': 3}},
 '2DBoxPlot.figure': {'digest': 'ac48d41ac185d8bb125597811c781181a8d0f82995309e89f497d6c95cd83537',
                      'length': 8928,
                      'props': ['2DBoxPlot.figure'],
                      'status': 200,
                      'traces': {'2DBoxPlot.figure': 3}},
 '2DBoxPlotEvalsSO.figure': {'digest': '0eab12303d6c3e278e5b9517a2c102a014e5da202d72f5206ecef1f2f7e91c9f',
                             'length': 8567,
                             'props': ['2DBoxPlotEvalsSO.figure'],
                             'status': 200,
                             'traces': {'2DBoxPlotEvalsSO.figure': 2}},
 '2DBoxPlotMO.figure': {'digest': '170e526d74935b7dc3e12dc0de6b0fcb307bca0ede39c88834d13cb5535ed30e',
                        'length': 7089,
                        'props': ['2DBoxPlotMO.figure'],
                        'status': 200,
                        'traces': {'2DBoxPlotMO.figure': 0}},
 '2DBoxPlotMisjudgementsSO.figure': {'digest': 'a750c983779ef630640e2f8d1052fb8af1056d7a9f973d4c6003fa7cbbf4ece8',
                                     'length': 8914,
                                     'props': ['2DBoxPlotMisjudgementsSO.figure'],
                                     'status': 200,
                                     'traces': {'2DBoxPlotMisjudgementsSO.figure': 3}},
 '2DBoxPlotPenaltySO.figure': {'digest': 'd1c4f3cb1ba17a5206c1bd28949d7ea39946e2c7167c0d81fe13afbdbc2145f2',
                               'length': 8922,
                               'props': ['2DBoxPlotPenaltySO.figure'],
                               'status': 200,
                               'traces': {'2DBoxPlotPenaltySO.figure': 3}},
 '2DLinePlot.figure': {'digest': '8c3874b02e5cad124d22dfd6b18f2407178debc3f5218f1ddfcc6d5f3b036140',
                       'length': 8013,
                       'props': ['2DLinePlot.figure'],
                       'status': 200,
                       'traces': {'2DLinePlot.figure': 3}},
 '2DLinePlotEvalsSO.figure': {'digest': '6025a27e8adf6cd5d95867653da9ee2201973e45f203dfafc61fddb74d04e794',
                              'length': 7738,
                              'props': ['2DLinePlotEvalsSO.figure'],
                              'status': 200,
                              'traces': {'2DLinePlotEvalsSO.figure': 2}},
 '2DLinePlotMO.figure': {'digest': '1cda20100b544dc8097b5a383617a23048018e79fd3e7de2e12c541a029dd79e',
                         'length': 7090,
                         'props': ['2DLinePlotMO.figure'],
                         'status': 200,
                         'traces': {'2DLinePlotMO.figure': 0}},
 '2DPlotTabContent.children': {'digest': 'fde343961819b9eb366d263785b879bc6c949e3c7b891859b5578a6c3ca8ca6d',
                               'length': 212,
                               'props': ['2DPlotTabContent.children'],
                               'status': 200,
                               'traces': {}},
 'LON_data.data': {'digest': 'e08ca5a27bc9265bff6e314ce09db8d5442e26667397c57c1641ea07ea297c4e',
                   'length': 784,
                   'props': ['LON_data.data'],
                   'status': 200,
                   'traces': {}},
 'STN_data.data': {'digest': 'cd68ca5e86e4cfa1fc61bfce82b8f9bf04de7f78f52a100bb43457052858d73a',
                   'length': 3201,
                   'props': ['STN_data.data'],
                   'status': 200,
                   'traces': {}},
 'axis-values.data': {'digest': 'fd4e8593326141cd34b4bf5b60f3deae7f1ef181c16d71675e84fc5fa950a323',
                      'length': 186,
                      'props': ['axis-values.data'],
                      'status': 200,
                      'traces': {}},
 'data-problem-specific.data': {'digest': '56f95ad8593d54dd850d57b3d35e42e1abf0f76ad91fa74cde7ff04a81b39bbb',
                                'length': 1581,
                                'props': ['data-problem-specific.data'],
                                'status': 200,
                                'traces': {}},
 'evals-mann-whitney-table.children': {'digest': 'f186f83de08481ab41db78708586533e28badf8de952742746ef8d86bdfb78a6',
                                       'length': 2766,
                                       'props': ['evals-mann-whitney-table.children'],
                                       'status': 200,
                                       'traces': {}},
 'evals-summary-table.children': {'digest': 'd288d416e5a138c0044e0b526a7b05dfb601728d767a3ef3c75efb0f0e6cd0d6',
                                  'length': 1628,
                                  'props': ['evals-summary-table.children'],
                                  'status': 200,
                                  'traces': {}},
 'experiment-description-display.children': {'digest': '7a43323fe4182989757004ca840cdf46b5b34e71d2e84530493c8ce4ec1f9209',
                                             'length': 413,
                                             'props': ['experiment-description-display.children'],
                                             'status': 200,
                                             'traces': {}},
 'hide-series-dropdown.options': {'digest': '3d404c2322b0298849bd1bc9b8df468204d07162cbc14f6c6ebafd9e4ff11cd2',
                                  'length': 190,
                                  'props': ['hide-series-dropdown.options'],
                                  'status': 200,
                                  'traces': {}},
 'knapsack-info-display.children': {'digest': 'cb39bb73dfbab4120f1e24a42a4f35cb92f0b75629d156496aa1d1028f843efd',
                                    'length': 4596,
                                    'props': ['knapsack-info-display.children'],
                                    'status': 200,
                                    'traces': {}},
 'lon-table-selected-pid-store.data': {'digest': '667e52ff463337a7a1ed609db6677b1d7625dee13b44c0274b76838155e6d171',
                                       'length': 86,
                                       'props': ['lon-table-selected-pid-store.data'],
                                       'status': 200,
                                       'traces': {}},
 'mann-whitney-table.children': {'digest': '1e25d70df3498630f92c24c8c67eafd76ff9f3ae68ce5aa960408dbf4a6b5ede',
                                 'length': 2739,
                                 'props': ['mann-whitney-table.children'],
                                 'status': 200,
                                 'traces': {}},
 'misjudgements-summary-table.children': {'digest': '7fcac64f28bf58c074d5cde34c6a4e51d61360c90f57a68ceaf43ee0aa4cbef8',
                                          'length': 1265,
                                          'props': ['misjudgements-summary-table.children'],
                                          'status': 200,
                                          'traces': {}},
 'penalty-filter-dropdown.options': {'digest': 'f6f108ebb53bf061e07f66224750c5c8c80ea880f38e52e4a7c3cf445479a8f7',
                                     'length': 123,
                                     'props': ['penalty-filter-dropdown.options'],
                                     'status': 200,
                                     'traces': {}},
 'penalty-filter-dropdown.value': {'digest': '64037ead88273a8e3b46e60d49b4ef6af890bcca747bfb269d77f980e00f735c',
                                   'length': 68,
                                   'props': ['penalty-filter-dropdown.value'],
                                   'status': 200,
                                   'traces': {}},
 'performance-summary-table.children': {'digest': '7f8b125455765950d3c6d637a13374ebdbadfcd40302d45c421d40f22d9d79db',
                                        'length': 1486,
                                        'props': ['performance-summary-table.children'],
                                        'status': 200,
                                        'traces': {}},
 'plotParetoFront.figure': {'digest': 'c12308331ff2ccfe84b9a2da8b3b4078b9cd0e470821c7ebe748f0db23156e75',
                            'length': 7037,
                            'props': ['plotParetoFront.figure'],
                            'status': 200,
                            'traces': {'plotParetoFront.figure': 0}},
 'plot_2d_data.data': {'digest': '0a27021e9f34d619642d060cf84eba960773fdd95052a483abc55bf7758fb6da',
                       'length': 12517,
                       'props': ['plot_2d_data.data'],
                       'status': 200,
                       'traces': {}},
 'plot_2d_data_table.data': {'digest': '6a577e552fff515bb13537998e7aa0e754bafa935faceb198185dfc8ca1dbefb',
                             'length': 12523,
                             'props': ['plot_2d_data_table.data'],
                             'status': 200,
                             'traces': {}},
 'print_STN_series_labels.children': {'digest': '3b1d15a9581c5a277a9038bb30abf926a217fe4eb2a37d744ca15c9097b0f24c',
                                      'length': 97,
                                      'props': ['print_STN_series_labels.children'],
                                      'status': 200,
                                      'traces': {}},
 'run-print-info.style': {'digest': '03a169bd7f143841057ca1b3c814d6e37ef3af394cc8ef2cbd93291690a51624',
                          'length': 105,
                          'props': ['run-print-info.style'],
                          'status': 200,
                          'traces': {}},
 'table1-selected-store.data': {'digest': '83f7607e05814529fc0e49c6b9a9a0cd74de8183cf0db907d156e8f45934492e',
                                'length': 64,
                                'props': ['table1-selected-store.data'],
                                'status': 200,
                                'traces': {}},
 'table1.data': {'digest': '2fc49b6625ab5f2f62a5a4f00fea4e8911684dd10319984ba4e84e829b857e0e',
                 'length': 213,
                 'props': ['table1.data'],
                 'status': 200,
                 'traces': {}},
 'table2-selected-output.children': {'digest': '873b2339264c91ffffdfddf345b45fa3995f3f0f078dd99d39b8ba06793233fe',
                                     'length': 423,
                                     'props': ['table2-selected-output.children'],
                                     'status': 200,
                                     'traces': {}},
 'table2-selected-store.data': {'digest': 'e9fd4f82bac258da5386f636d38b88396f5c43d905366ab51387f692d83bc9e2',
                                'length': 64,
                                'props': ['table2-selected-store.data'],
                                'status': 200,
                                'traces': {}},
 'table2.data': {'digest': 'b2a3056a9e21b624483921021bf62d7ec31c7ec8ccddd3d9ac19210091786480',
                 'length': 2670,
                 'props': ['table2.data'],
                 'status': 200,
                 'traces': {}}}

CHAIN = {'..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': {'digest': '187231ddee9a22c9a4ad3542f6027a92cec16d11508c9f82c50d81388ee094af',
                                                                                                                                                        'length': 1287,
                                                                                                                                                        'props': ['MO_data_PPP.data',
                                                                                                                                                                  'STN_MO_data.data',
                                                                                                                                                                  'STN_MO_series_labels.data',
                                                                                                                                                                  'STN_data_processed.data',
                                                                                                                                                                  'STN_series_labels.data',
                                                                                                                                                                  'noisy_fitnesses_data.data'],
                                                                                                                                                        'status': 200,
                                                                                                                                                        'traces': {}},
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': {'digest': 'e8020fbd3d15bc0869ed50644fa2516952bb9f11cbfa0d550f4fab54cfbdbc4c',
                                                                       'length': 161,
                                                                       'props': ['PID.data',
                                                                                 'fit_func_store.data',
                                                                                 'opt_goal.data',
                                                                                 'optimum.data'],
                                                                       'status': 200,
                                                                       'traces': {}},
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': {'digest': 'b972b1a6b94c2b2615ddf4566507ba87135e890afc4648d62ed64e3c2d31a8b0',
                                                                                                                                                                                                                           'length': 23610,
                                                                                                                                                                                                                           'props': ['lon-feas-error-correlations.children',
                                                                                                                                                                                                                                     'lon-feas-error-scatter.figure',
                                                                                                                                                                                                                                     'lon-selected-correlation.children',
                                                                                                                                                                                                                                     'lon-stats-table.children',
                                                                                                                                                                                                                                     'run-print-info.children',
                                                                                                                                                                                                                                     'stn-stats-table.children',
                                                                                                                                                                                                                                     'trajectory-plot.figure'],
                                                                                                                                                                                                                           'status': 200,
                                                                                                                                                                                                                           'traces': {'lon-feas-error-scatter.figure': 0,
                                                                                                                                                                                                                                      'trajectory-plot.figure': 17}},
 'LON_data.data': {'digest': 'e08ca5a27bc9265bff6e314ce09db8d5442e26667397c57c1641ea07ea297c4e',
                   'length': 784,
                   'props': ['LON_data.data'],
                   'status': 200,
                   'traces': {}},
 'STN_data.data': {'digest': 'cd68ca5e86e4cfa1fc61bfce82b8f9bf04de7f78f52a100bb43457052858d73a',
                   'length': 3201,
                   'props': ['STN_data.data'],
                   'status': 200,
                   'traces': {}},
 'data-problem-specific.data': {'digest': '56f95ad8593d54dd850d57b3d35e42e1abf0f76ad91fa74cde7ff04a81b39bbb',
                                'length': 1581,
                                'props': ['data-problem-specific.data'],
                                'status': 200,
                                'traces': {}},
 'plot_2d_data.data': {'digest': '0a27021e9f34d619642d060cf84eba960773fdd95052a483abc55bf7758fb6da',
                       'length': 12517,
                       'props': ['plot_2d_data.data'],
                       'status': 200,
                       'traces': {}}}

TABLES = {'constants': {'DISPLAY1_COLUMNS': ['problem_type',
                                    'problem_goal',
                                    'problem_name',
                                    'dimensions',
                                    'opt_global',
                                    'fit_func',
                                    'PID'],
               'DISPLAY2_DEDUP_KEYS': ['PID', 'algo_type', 'algo_name', 'noise', 'fit_func'],
               'DISPLAY2_DROP_COLUMNS': ['n_gens',
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
                                         'run_id',
                                         'parent_run_id',
                                         'payload_path',
                                         'peak_ram_mb',
                                         'penalty'],
               'DISPLAY2_HIDDEN_COLUMNS': ['problem_type',
                                           'problem_goal',
                                           'problem_name',
                                           'dimensions',
                                           'opt_global',
                                           'PID',
                                           'experiment_name',
                                           'experiment_description'],
               'LIST_COLUMNS': ['rep_sols',
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
                                'n_gens_pareto_best'],
               'LON_HIDDEN_COLUMNS': ['problem_name',
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
                                      'visit_proportions']},
 'df_no_lists': {'columns': ['PID',
                             'problem_type',
                             'problem_goal',
                             'problem_name',
                             'dimensions',
                             'opt_global',
                             'fit_func',
                             'experiment_name',
                             'experiment_description',
                             'algo_type',
                             'algo_name',
                             'noise',
                             'seed',
                             'seed_signature',
                             'n_gens',
                             'n_evals',
                             'stop_trigger',
                             'n_unique_sols',
                             'final_fit',
                             'max_fit',
                             'min_fit',
                             'penalty',
                             'peak_ram_mb',
                             'evals_to_best',
                             'evals_to_final',
                             'evals_to_best_noisy',
                             'evals_to_final_noisy',
                             'final_fit_noisy',
                             'max_fit_noisy',
                             'min_fit_noisy',
                             'n_misjudgements',
                             'n_increasing_noise',
                             'n_comparison_misjudgements',
                             'n_constraint_misjudgements'],
                 'digest': 'ca2a66fea01306bfdf8b63b7d20a6a10cabf30410b1a3be78bc4da731d745260',
                 'dtypes': ['object',
                            'object',
                            'object',
                            'object',
                            'int64',
                            'float64',
                            'object',
                            'object',
                            'object',
                            'object',
                            'object',
                            'float64',
                            'int64',
                            'object',
                            'int64',
                            'int64',
                            'object',
                            'int64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'float64',
                            'int64',
                            'int64',
                            'int64',
                            'int64'],
                 'shape': [17, 34]},
 'display1_df': {'columns': ['problem_type',
                             'problem_goal',
                             'problem_name',
                             'dimensions',
                             'opt_global',
                             'fit_func',
                             'PID'],
                 'digest': 'edc4c357a25a76aea3a551c934945f3fca8d579cf9e4a00ecccf7737713d3e93',
                 'dtypes': ['object', 'object', 'object', 'int64', 'float64', 'object', 'object'],
                 'shape': [2, 7]},
 'display2_df': {'columns': ['PID',
                             'problem_type',
                             'problem_goal',
                             'problem_name',
                             'dimensions',
                             'opt_global',
                             'fit_func',
                             'experiment_name',
                             'experiment_description',
                             'algo_type',
                             'algo_name',
                             'noise',
                             'no_runs'],
                 'digest': '02c64ce3725cb1a16a50acc94283c6a4f9299321d407912e54ed64d55792d30e',
                 'dtypes': ['object',
                            'object',
                            'object',
                            'object',
                            'int64',
                            'float64',
                            'object',
                            'object',
                            'object',
                            'object',
                            'object',
                            'float64',
                            'int64'],
                 'shape': [9, 13]},
 'lon_display_columns': ['PID', 'fit_func', 'algo_name', 'noise', 'n_runs', 'iterations'],
 'step_indices': {'comparison_max': [2],
                  'comparison_min': [],
                  'constraint': [3],
                  'increasing_noise': [2]}}

IMPORT_SURFACE = {'noisyvis_modules': ['noisyvis',
                      'noisyvis.algorithms',
                      'noisyvis.algorithms.multi_objective',
                      'noisyvis.algorithms.operators',
                      'noisyvis.algorithms.single_objective',
                      'noisyvis.analysis',
                      'noisyvis.analysis.graph_stats',
                      'noisyvis.common',
                      'noisyvis.common.distance',
                      'noisyvis.common.embedding',
                      'noisyvis.common.geometry',
                      'noisyvis.dashboard',
                      'noisyvis.dashboard.Dashboard',
                      'noisyvis.dashboard.DashboardHelpers',
                      'noisyvis.dashboard.components',
                      'noisyvis.dashboard.layout',
                      'noisyvis.dashboard.layout.components',
                      'noisyvis.dashboard.layout.main_layout',
                      'noisyvis.dashboard.layout.stores',
                      'noisyvis.dashboard.layout.styles',
                      'noisyvis.dataio',
                      'noisyvis.dataio.column_config',
                      'noisyvis.dataio.transformers',
                      'noisyvis.problems',
                      'noisyvis.problems.constraints',
                      'noisyvis.problems.continuous',
                      'noisyvis.problems.instances',
                      'noisyvis.problems.jump',
                      'noisyvis.problems.knapsack',
                      'noisyvis.problems.knapsack_mo',
                      'noisyvis.problems.onemax',
                      'noisyvis.results',
                      'noisyvis.results.paths',
                      'noisyvis.results.store',
                      'noisyvis.tracking',
                      'noisyvis.tracking.logger',
                      'noisyvis.viz',
                      'noisyvis.viz.config',
                      'noisyvis.viz.graph',
                      'noisyvis.viz.graph.lon',
                      'noisyvis.viz.graph.stn',
                      'noisyvis.viz.layout',
                      'noisyvis.viz.plots',
                      'noisyvis.viz.plots.base',
                      'noisyvis.viz.plots.lon_stats',
                      'noisyvis.viz.plots.pareto',
                      'noisyvis.viz.plots.pareto.analysis',
                      'noisyvis.viz.plots.pareto.animation',
                      'noisyvis.viz.plots.pareto.basic',
                      'noisyvis.viz.plots.pareto.correlation',
                      'noisyvis.viz.plots.pareto.noisy',
                      'noisyvis.viz.plots.pareto.subplots',
                      'noisyvis.viz.plots.performance',
                      'noisyvis.viz.plots.performance.box_plots',
                      'noisyvis.viz.plots.performance.line_plots',
                      'noisyvis.viz.plots.registry',
                      'noisyvis.viz.styling',
                      'noisyvis.viz.traces'],
 'statements': {'analysis/__init__.py': [],
                'analysis/graph_stats.py': [['from', 'typing', 'Dict', None],
                                            ['from', 'typing', 'List', None],
                                            ['from', 'typing', 'Any', None],
                                            ['from', 'typing', 'Optional', None],
                                            ['from', 'typing', 'Tuple', None],
                                            ['from', 'dataclasses', 'dataclass', None],
                                            ['import', 'networkx', 'nx'],
                                            ['import', 'numpy', 'np'],
                                            ['from', 'noisyvis.common', 'lookup_map', None]],
                'app/app.py': [['import', 'dash', None],
                               ['from', 'dash', 'html', None],
                               ['from', 'dash', 'dcc', None]],
                'app/pages/mlflow_browser.py': [['from', '__future__', 'annotations', None],
                                                ['from', 'pathlib', 'Path', None],
                                                ['import', 'pandas', 'pd'],
                                                ['from', 'dash', 'html', None],
                                                ['from', 'dash', 'dash_table', None],
                                                ['from', 'dash', 'dcc', None],
                                                ['from', 'dash', 'Input', None],
                                                ['from', 'dash', 'Output', None],
                                                ['from', 'dash', 'State', None],
                                                ['from', 'dash', 'callback', None],
                                                ['import', 'dash', None],
                                                ['import', 'mlflow', None],
                                                ['from',
                                                 'noisyvis.results.mlflow_query',
                                                 'list_experiments_df',
                                                 None],
                                                ['from',
                                                 'noisyvis.results.mlflow_query',
                                                 'list_runs_df',
                                                 None],
                                                ['from',
                                                 'noisyvis.results.mlflow_query',
                                                 'select_present_columns',
                                                 None]],
                'dashboard/Dashboard.py': [['import', 'math', None],
                                           ['import', 'dash', None],
                                           ['from', 'dash', 'html', None],
                                           ['from', 'dash', 'dcc', None],
                                           ['from', 'dash', 'dash_table', None],
                                           ['from', 'dash', 'Input', None],
                                           ['from', 'dash', 'Output', None],
                                           ['from', 'dash', 'State', None],
                                           ['from', 'dash', 'ctx', None],
                                           ['import', 'pandas', 'pd'],
                                           ['import', 'matplotlib.pyplot', 'plt'],
                                           ['import', 'plotly.graph_objects', 'go'],
                                           ['import', 'plotly.express', 'px'],
                                           ['import', 'networkx', 'nx'],
                                           ['import', 'numpy', 'np'],
                                           ['from', 'sklearn.manifold', 'MDS', 'MDS_sklearn'],
                                           ['from', 'sklearn.manifold', 'ClassicalMDS', None],
                                           ['from', 'sklearn.manifold', 'TSNE', None],
                                           ['from', 'collections', 'defaultdict', None],
                                           ['from',
                                            'concurrent.futures',
                                            'ThreadPoolExecutor',
                                            None],
                                           ['from', 'dataclasses', 'replace', 'dataclass_replace'],
                                           ['import', 'os', None],
                                           ['from',
                                            'noisyvis.dashboard.DashboardHelpers',
                                            '*',
                                            None],
                                           ['from', 'noisyvis.problems.onemax', '*', None],
                                           ['from', 'noisyvis.problems.jump', '*', None],
                                           ['from', 'noisyvis.problems.knapsack', '*', None],
                                           ['from', 'noisyvis.problems.continuous', '*', None],
                                           ['from',
                                            'noisyvis.problems.instances',
                                            'load_problem_KP',
                                            None],
                                           ['from',
                                            'noisyvis.problems.instances',
                                            'get_knapsack_problem_stats',
                                            None],
                                           ['from',
                                            'noisyvis.problems.instances',
                                            'interpret_correlation',
                                            None],
                                           ['from', 'noisyvis.common.embedding', '*', None],
                                           ['from',
                                            'noisyvis.dashboard.layout',
                                            'create_layout',
                                            None],
                                           ['from', 'noisyvis.dashboard.layout', 'TAB_STYLE', None],
                                           ['from',
                                            'noisyvis.dashboard.layout',
                                            'TAB_SELECTED_STYLE',
                                            None],
                                           ['from',
                                            'noisyvis.dashboard.layout',
                                            '_build_schematic_figure',
                                            None],
                                           ['from',
                                            'noisyvis.dashboard.layout',
                                            '_build_schematic_legend',
                                            None],
                                           ['from',
                                            'noisyvis.dashboard.layout.stores',
                                            'LON_TABLE_SELECTED_PID_STORE',
                                            None],
                                           ['from', 'noisyvis.viz', 'parse_callback_inputs', None],
                                           ['from', 'noisyvis.viz', 'PlotConfig', None],
                                           ['from',
                                            'noisyvis.viz',
                                            'generate_run_summary_string',
                                            None],
                                           ['from', 'noisyvis.viz', 'add_stn_trajectories', None],
                                           ['from', 'noisyvis.viz', 'add_mo_fronts', None],
                                           ['from', 'noisyvis.viz', 'add_prior_noise_stn_v4', None],
                                           ['from', 'noisyvis.viz', 'add_prior_noise_stn_v5', None],
                                           ['from',
                                            'noisyvis.viz',
                                            'add_prior_noise_stn_algo_pov',
                                            None],
                                           ['from', 'noisyvis.viz', 'add_lon_nodes', None],
                                           ['from', 'noisyvis.viz', 'add_lon_edges', None],
                                           ['from', 'noisyvis.viz', 'debug_mo_counts', None],
                                           ['from', 'noisyvis.viz', 'style_nodes', None],
                                           ['from', 'noisyvis.viz', 'calculate_positions', None],
                                           ['from',
                                            'noisyvis.viz',
                                            'LON_SCATTER_AXIS_LABELS',
                                            None],
                                           ['from',
                                            'noisyvis.viz',
                                            'LON_SCATTER_DEFAULT_X_AXIS',
                                            None],
                                           ['from',
                                            'noisyvis.viz',
                                            'LON_SCATTER_DEFAULT_Y_AXIS',
                                            None],
                                           ['from',
                                            'noisyvis.viz',
                                            'LON_SCATTER_DEFAULT_PLOT_STYLE',
                                            None],
                                           ['from', 'noisyvis.viz', 'plot_lon_stats', None],
                                           ['from', 'noisyvis.viz', 'plot_lon_stats_multi', None],
                                           ['from', 'noisyvis.viz', 'build_all_traces', None],
                                           ['from', 'noisyvis.viz', 'create_guide_traces', None],
                                           ['from', 'noisyvis.viz', 'create_axis_settings', None],
                                           ['from', 'noisyvis.viz', 'create_figure', None],
                                           ['from',
                                            'noisyvis.analysis.graph_stats',
                                            'calculate_lon_statistics',
                                            None],
                                           ['from',
                                            'noisyvis.analysis.graph_stats',
                                            'compute_node_feasibility_error',
                                            None],
                                           ['from',
                                            'noisyvis.analysis.graph_stats',
                                            'compute_pairwise_correlations',
                                            None],
                                           ['from',
                                            'noisyvis.analysis.graph_stats',
                                            'compute_correlation_pair',
                                            None],
                                           ['from',
                                            'noisyvis.dashboard.components',
                                            'build_correlation_table',
                                            None],
                                           ['from',
                                            'noisyvis.dashboard.components',
                                            'build_selected_correlation_display',
                                            None],
                                           ['from',
                                            'noisyvis.common',
                                            'is_continuous_solution',
                                            None],
                                           ['from', 'noisyvis.viz.plots', 'get_pareto_plot', None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_line',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_line_mo',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box_mo',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_line_evals',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box_evals',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box_penalty',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box_misjudgements_so',
                                            None],
                                           ['from',
                                            'noisyvis.viz.plots.performance',
                                            'plot2d_box_advanced_misjudgements_so',
                                            None],
                                           ['from', 'noisyvis.dataio', 'DashboardData', None],
                                           ['from',
                                            'noisyvis.dataio',
                                            'DISPLAY2_HIDDEN_COLUMNS',
                                            None],
                                           ['from', 'noisyvis.dataio', 'LON_HIDDEN_COLUMNS', None],
                                           ['from',
                                            'noisyvis.dataio.transformers',
                                            'create_display2_df',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.transformers',
                                            'increasing_noise_step_indices',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.transformers',
                                            'comparison_misjudgement_step_indices',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.transformers',
                                            'constraint_misjudgement_step_indices',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'DISPLAY1_COLUMNS',
                                            None]],
                'dashboard/DashboardHelpers.py': [['import', 'numpy', 'np'],
                                                  ['import', 'math', None],
                                                  ['from',
                                                   'noisyvis.viz.plots.performance',
                                                   'plot2d_line',
                                                   None],
                                                  ['from',
                                                   'noisyvis.viz.plots.performance',
                                                   'plot2d_box',
                                                   None],
                                                  ['from',
                                                   'noisyvis.viz.plots.performance',
                                                   'plot2d_line_mo',
                                                   None],
                                                  ['from',
                                                   'noisyvis.viz.plots.performance',
                                                   'plot2d_box_mo',
                                                   None],
                                                  ['from',
                                                   'noisyvis.common',
                                                   'hamming_distance',
                                                   None]],
                'dashboard/__init__.py': [['from',
                                           'noisyvis.dashboard.DashboardHelpers',
                                           '*',
                                           None]],
                'dashboard/components.py': [['from', 'typing', 'Any', None],
                                            ['from', 'typing', 'Dict', None],
                                            ['from', 'typing', 'List', None],
                                            ['from', 'dash', 'dash_table', None]],
                'dashboard/layout/__init__.py': [['from',
                                                  'noisyvis.dashboard.layout.main_layout',
                                                  'create_layout',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.components',
                                                  '_build_schematic_figure',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.components',
                                                  '_build_schematic_legend',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.styles',
                                                  'TAB_STYLE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.styles',
                                                  'TAB_SELECTED_STYLE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.styles',
                                                  'SECTION_STYLE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.styles',
                                                  'SELECTION_OUTPUT_STYLE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'TABLE1_SELECTED_STORE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'TABLE2_SELECTED_STORE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'DATA_PROBLEM_SPECIFIC',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'OPTIMUM_STORE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'PID_STORE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'OPT_GOAL_STORE',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'PLOT_2D_DATA',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'STN_DATA',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'LON_DATA',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'STN_DATA_PROCESSED',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'STN_SERIES_LABELS',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'NOISY_FITNESSES_DATA',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'AXIS_VALUES',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'STN_MO_DATA',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'STN_MO_SERIES_LABELS',
                                                  None],
                                                 ['from',
                                                  'noisyvis.dashboard.layout.stores',
                                                  'MO_DATA_PPP',
                                                  None]],
                'dashboard/layout/components.py': [['import', 'plotly.graph_objects', 'go'],
                                                   ['from', 'dash', 'html', None],
                                                   ['from', 'dash', 'dcc', None],
                                                   ['from', 'dash', 'dash_table', None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'TAB_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'TAB_SELECTED_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'SECTION_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'SELECTION_OUTPUT_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'DROPDOWN_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'DROPDOWN_STYLE_WITH_MARGIN',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'SMALL_INPUT_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'INLINE_BLOCK_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'INLINE_BLOCK_WITH_MARGIN',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'FLEX_ROW_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'FLEX_WITH_GAP_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'INLINE_DROPDOWN_WRAPPER_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'INLINE_VERTICAL_ALIGN_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'INLINE_VERTICAL_ALIGN_NO_MARGIN_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'FULL_WIDTH_INLINE_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.dashboard.layout.styles',
                                                    'MONOSPACE_STYLE',
                                                    None],
                                                   ['from',
                                                    'noisyvis.viz',
                                                    'LON_SCATTER_AXIS_OPTIONS',
                                                    None],
                                                   ['from',
                                                    'noisyvis.viz',
                                                    'LON_SCATTER_DEFAULT_X_AXIS',
                                                    None],
                                                   ['from',
                                                    'noisyvis.viz',
                                                    'LON_SCATTER_DEFAULT_Y_AXIS',
                                                    None],
                                                   ['from',
                                                    'noisyvis.viz',
                                                    'LON_SCATTER_PLOT_STYLE_OPTIONS',
                                                    None],
                                                   ['from',
                                                    'noisyvis.viz',
                                                    'LON_SCATTER_DEFAULT_PLOT_STYLE',
                                                    None]],
                'dashboard/layout/main_layout.py': [['from', 'dash', 'html', None],
                                                    ['from', 'dash', 'dcc', None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.stores',
                                                     'create_all_stores',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_schematic_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_problem_selection_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_2d_plot_tabs',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_round_stats_checkbox',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_misjudgements_summary_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_performance_summary_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_mann_whitney_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_evals_summary_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_evals_mann_whitney_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_algorithm_table',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_pareto_front_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_multiobjective_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_stn_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_lon_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_plot_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_opacity_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_axis_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_annotation_options_section',
                                                     None],
                                                    ['from',
                                                     'noisyvis.dashboard.layout.components',
                                                     'create_main_plot_section',
                                                     None]],
                'dashboard/layout/stores.py': [['from', 'dash', 'dcc', None]],
                'dashboard/layout/styles.py': [],
                'dataio/__init__.py': [['from', 'dataclasses', 'dataclass', None],
                                       ['from', 'typing', 'List', None],
                                       ['import', 'pandas', 'pd'],
                                       ['from', 'noisyvis.results.paths', 'WAREHOUSE_DIR', None],
                                       ['from',
                                        'noisyvis.results.store',
                                        'load_algo_results',
                                        None],
                                       ['from', 'noisyvis.results.store', 'load_lon_results', None],
                                       ['from', 'noisyvis.results.store', 'DataLoadError', None],
                                       ['from',
                                        'noisyvis.dataio.transformers',
                                        'create_df_no_lists',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.transformers',
                                        'create_display1_df',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.transformers',
                                        'create_display2_df',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.transformers',
                                        'get_lon_display_columns',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'LON_HIDDEN_COLUMNS',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'DISPLAY2_HIDDEN_COLUMNS',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'DISPLAY1_COLUMNS',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'LIST_COLUMNS',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'DISPLAY2_DROP_COLUMNS',
                                        None],
                                       ['from',
                                        'noisyvis.dataio.column_config',
                                        'DISPLAY2_DEDUP_KEYS',
                                        None]],
                'dataio/column_config.py': [],
                'dataio/transformers.py': [['import', 'numpy', 'np'],
                                           ['import', 'pandas', 'pd'],
                                           ['from', 'typing', 'List', None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'LIST_COLUMNS',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'DISPLAY1_COLUMNS',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'DISPLAY2_DROP_COLUMNS',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'DISPLAY2_DEDUP_KEYS',
                                            None],
                                           ['from',
                                            'noisyvis.dataio.column_config',
                                            'LON_HIDDEN_COLUMNS',
                                            None]]},
 'top_level_packages': ['IPython',
                        'PIL',
                        'abc',
                        'argparse',
                        'array',
                        'ast',
                        'asttokens',
                        'asyncio',
                        'atexit',
                        'backports',
                        'base64',
                        'bdb',
                        'binascii',
                        'bisect',
                        'blinker',
                        'brotli',
                        'builtins',
                        'bz2',
                        'cProfile',
                        'calendar',
                        'certifi',
                        'charset_normalizer',
                        'click',
                        'cloudpickle',
                        'cmath',
                        'cmd',
                        'code',
                        'codecs',
                        'codeop',
                        'collections',
                        'colorama',
                        'colorlog',
                        'colorsys',
                        'comm',
                        'concurrent',
                        'configparser',
                        'contextlib',
                        'contextvars',
                        'copy',
                        'copyreg',
                        'csv',
                        'ctypes',
                        'cuda',
                        'curses',
                        'cycler',
                        'cython_runtime',
                        'dash',
                        'dataclasses',
                        'datetime',
                        'dateutil',
                        'deap',
                        'decimal',
                        'decorator',
                        'difflib',
                        'dis',
                        'email',
                        'encodings',
                        'enum',
                        'errno',
                        'executing',
                        'faulthandler',
                        'fcntl',
                        'filecmp',
                        'fileinput',
                        'flask',
                        'fnmatch',
                        'fractions',
                        'functools',
                        'gc',
                        'genericpath',
                        'getopt',
                        'getpass',
                        'gettext',
                        'glob',
                        'grp',
                        'gzip',
                        'hashlib',
                        'heapq',
                        'hmac',
                        'html',
                        'http',
                        'idna',
                        'importlib',
                        'importlib_metadata',
                        'inspect',
                        'io',
                        'ipaddress',
                        'ipykernel',
                        'itertools',
                        'itsdangerous',
                        'jedi',
                        'jinja2',
                        'joblib',
                        'json',
                        'jupyter_client',
                        'jupyter_core',
                        'keyword',
                        'kiwisolver',
                        'linecache',
                        'locale',
                        'logging',
                        'lzma',
                        'markupsafe',
                        'marshal',
                        'math',
                        'matplotlib',
                        'mimetypes',
                        'mmap',
                        'mpl_toolkits',
                        'multiprocessing',
                        'nest_asyncio',
                        'networkx',
                        'noisyvis',
                        'ntpath',
                        'numbers',
                        'numpy',
                        'opcode',
                        'operator',
                        'optuna',
                        'os',
                        'packaging',
                        'pandas',
                        'parso',
                        'pathlib',
                        'pdb',
                        'pickle',
                        'pkgutil',
                        'platform',
                        'platformdirs',
                        'plistlib',
                        'plotly',
                        'posix',
                        'posixpath',
                        'pprint',
                        'profile',
                        'prompt_toolkit',
                        'pstats',
                        'psutil',
                        'pure_eval',
                        'pwd',
                        'pyarrow',
                        'pydoc',
                        'pydoc_data',
                        'pyexpat',
                        'pygments',
                        'pyparsing',
                        'pytz',
                        'queue',
                        'quopri',
                        'random',
                        're',
                        'reprlib',
                        'requests',
                        'resource',
                        'retrying',
                        'runpy',
                        'scipy',
                        'secrets',
                        'select',
                        'selectors',
                        'shlex',
                        'shutil',
                        'signal',
                        'site',
                        'six',
                        'sklearn',
                        'socket',
                        'socketserver',
                        'socks',
                        'sqlite3',
                        'ssl',
                        'stack_data',
                        'stat',
                        'statistics',
                        'string',
                        'stringprep',
                        'struct',
                        'subprocess',
                        'sys',
                        'sysconfig',
                        'tarfile',
                        'tempfile',
                        'termios',
                        'textwrap',
                        'threading',
                        'threadpoolctl',
                        'time',
                        'timeit',
                        'token',
                        'tokenize',
                        'tornado',
                        'tqdm',
                        'traceback',
                        'traitlets',
                        'types',
                        'typing',
                        'typing_extensions',
                        'unicodedata',
                        'unittest',
                        'urllib',
                        'urllib3',
                        'uuid',
                        'warnings',
                        'wcwidth',
                        'weakref',
                        'webbrowser',
                        'werkzeug',
                        'xml',
                        'zipfile',
                        'zipimport',
                        'zipp',
                        'zlib',
                        'zmq',
                        'zoneinfo']}

MLFLOW = {'config': {'assets_folder': 'src/noisyvis/app/assets',
            'pages_folder': 'src/noisyvis/app/pages',
            'use_pages': True},
 'dependencies': ['..tracking-uri.children...experiments-table-wrap.children..',
                  'runs-table-wrap.children',
                  'selected-run-ids.data',
                  '.._pages_content.children..._pages_store.data..',
                  '_pages_dummy.children'],
 'dependencies_digest': 'e40392fd4f045b2325b89837a4400b11ae87c4da9c26fafd72440563d88481ee',
 'index_status': 200,
 'layout_digest': 'ef453062c1e4b1a302c308cb7bd85e01ebd1828b696f0107c6b88ffe531834e2',
 'module': 'noisyvis.app.app',
 'page_file': 'app/pages/mlflow_browser.py',
 'page_globals': ['EXP_COLS',
                  'Input',
                  'Output',
                  'Path',
                  'RUN_COLS_DEFAULT',
                  'State',
                  'annotations',
                  'callback',
                  'dash',
                  'dash_table',
                  'dcc',
                  'html',
                  'layout',
                  'list_experiments_df',
                  'list_runs_df',
                  'mlflow',
                  'pd',
                  'remember_runs',
                  'render_experiments',
                  'render_runs_table',
                  'select_present_columns'],
 'registry': {'pages.mlflow_browser': {'module': 'pages.mlflow_browser',
                                       'name': 'Experiments',
                                       'path': '/experiments',
                                       'relative_path': '/experiments'}}}

STATEMENTS = {'app/app.py': [{'ast': '83bbd18258b8f5aa841b8d4d088a2dca8eed99d62450542a9136f9e4a64d3004',
                 'key': ['assign', 'app'],
                 'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                 'module': 'app/app.py',
                 'name': None},
                {'ast': 'c6e41a315906cd0cd0bc11dde2aff463c92580c6ec60ac7113b5cb8d69ee8fe8',
                 'key': ['assign', 'app.layout'],
                 'literals': '7543ad1bd3b91e9efb96b1fa0326db65ee26d15ef1d3b17dd364d76ca2dc7a4d',
                 'module': 'app/app.py',
                 'name': None},
                {'ast': '48167a92027cc18d3015dea506050c537340dbdb14865abbf14915b81f2322a5',
                 'key': ['main'],
                 'literals': 'af3f4b5db7c7166abef951cf198d6a6a67c7f32ae843ef7e4812e0a2b1ec9b70',
                 'module': 'app/app.py',
                 'name': None}],
 'app/pages/mlflow_browser.py': [{'ast': '5084a1268432a309c2087beda5f335365ad119566a506090ca32a2945cca05b5',
                                  'key': ['expr',
                                          "dash.register_page(__name__, path='/experiments', "
                                          "name='Experiments')"],
                                  'literals': '8464153a08fc3a7ea6925d027a14823d5b29d9e0a1af8b7e24a9e133e6e51fb1',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': None},
                                 {'ast': 'c29d4e0535bfb64a637c28117446c74c1ee727afbf85eca95b1dc06641b4e91e',
                                  'key': ['assign', 'layout'],
                                  'literals': 'b98a90bd73305af7dfbd67d994e49edb0e2dfe95196d30259c00921773f573e9',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': None},
                                 {'ast': '249c6e26e730b780a814ca089c9ce2dcd7653668639c9d4edfd718e85c950433',
                                  'key': ['assign', 'EXP_COLS'],
                                  'literals': '3caecd94cad6d17cd1959bf3fc3bf207c8910c6af387698a5fe6858f5014d17d',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': None},
                                 {'ast': '16128d3ab71d478f7cd2c3d9e070adbc0d287a0e412bb0dae5527bd4f0913491',
                                  'key': ['assign', 'RUN_COLS_DEFAULT'],
                                  'literals': '80ac0f93a02a7bc8eb6085cb1cc668ca3de1f37b47c6300900ca944a08c17037',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': None},
                                 {'ast': 'bb5be7a162f954f5996115b08d0afbe81e9c7f0a34f848d376fb34636c232df8',
                                  'key': ['def', 'render_experiments'],
                                  'literals': '934500ffd2b202cf7d0d5ec4276ac30a1c1b5928ffe8ac2b2af2c75e9195b36b',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': 'render_experiments'},
                                 {'ast': '8aeab43cfc49526cffa6420ee0492bb3b624a637105e0372bed2df82570bccf8',
                                  'key': ['def', 'render_runs_table'],
                                  'literals': 'e9aa72abc6af025b22e3989926b7d9e12a8673d7633fac281048c3cabb46fc25',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': 'render_runs_table'},
                                 {'ast': '1c8818988a12d927f8cafa7bd4e0961a6df0b608b7b447c03fe09c9143e6454f',
                                  'key': ['def', 'remember_runs'],
                                  'literals': '20ed6086735e72779d6138ee8e827000808e6663f139d25a96464ad679ef1967',
                                  'module': 'app/pages/mlflow_browser.py',
                                  'name': 'remember_runs'}],
 'dashboard/Dashboard.py': [{'ast': '5895eade891ff979d70d320e20dae6ad54291798779fefd97ee372d6f58afc50',
                             'key': ['assign', 'data'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': 'df383798dcf265cd22ddcfe121baf4b1000689a131612eaf9242e89bb3589463',
                             'key': ['assign', 'df'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': 'dd22ad7ec2d9c063366df379a4c9f2b98e3cb84c2dae608b6570a81bf8d9f349',
                             'key': ['assign', 'df_LONs'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': 'c1fca2593f77dac67eeeeea0a47c76a950a5ce3509ec508b6defac498a0671a8',
                             'key': ['assign', 'df_no_lists'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '1b092668e138e6e106666babe3208106278fe01ab50561412d6e8cb952d94d9f',
                             'key': ['assign', 'display1_df'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '1e92fcca030e6651ce8fc02e02e6f2ebda97189b8ae96aa1fbd121009e00947a',
                             'key': ['assign', 'display2_df'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '7ad4ecc4339bdf8e5b935d7d0869b6556205d7c94cfb3c50df79a6114cbcb6a3',
                             'key': ['assign', 'LON_display_columns'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '0d3396d950f39363cb0412935ae563541da09fd0124431390480613dfb95cd3a',
                             'key': ['assign', 'display2_hidden_cols'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '9167e8ea03ca53768786e6171156ab0421699ac2eaacf1f0dc404d7c62a0a36b',
                             'key': ['assign', 'LON_hidden_cols'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '52f86199797989922e407178afe622e9167be9d5b2dbe60b9884032bc17798e2',
                             'key': ['assign', 'experiment_names'],
                             'literals': '9733f31af9da0377d1712ab3ea452d6dd34ac4a0cb85f938817955e326c43b72',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '6f34d7e159883e96cd7f8f3dd805d9503e3405c3b8b1e06bc1a590b94eb9bed4',
                             'key': ['assign', 'experiment_descriptions'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '653a5efe65b9f99641cc635f978a2b38c531cdfda4ad5fa8aed8108c11d3034b',
                             'key': ['if',
                                     "'experiment_description' in df.columns and 'experiment_name' "
                                     'in df.columns'],
                             'literals': '3e1448f0d6e69476bc24127945baad6153b9add90d69ca006cf54b4e245e6f7a',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '9d434d28e3dfd80dbfd760764b502f98d54060785855ada5d1a8b71e12db9642',
                             'key': ['def', '_filter_by_experiment'],
                             'literals': 'bcaf7f02e45d58c536468843e72eef87f5f84bd12eb0a9769f3bb1d69a5c17f6',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_filter_by_experiment'},
                            {'ast': '5a4991aa3826cbc58cc3c04cd8a4a85b73cdd5cb224e561821f1ddf8e21c43f3',
                             'key': ['def', '_add_guide_nodes'],
                             'literals': 'f4aa2ca612f1e256a4fea74527195ad3c4ba26960cce2e00ee3b0f8d2ec06056',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_add_guide_nodes'},
                            {'ast': '4064c4afd209aacff5f6a7fda2b35b043843489785510c5c55d7c35d169e683c',
                             'key': ['assign', 'app'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': 'f7cda1edb09c99b6a11cba272a56c22d7edbcec44193b0320656e70a902c2079',
                             'key': ['assign', 'tab_style'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '11cda325fd040268c75e963f31cb304c17b68a7b30aa3cd48774e1b11bf20357',
                             'key': ['assign', 'tab_selected_style'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '9b79012a2584e54d2351b9fad07d0868fdc4b6e7750149629330b9e9fb77fc72',
                             'key': ['assign', 'app.layout'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '08546b3eec6dba5b19c4e322f4d8876903983d07211263702ff3341d3076651b',
                             'key': ['assign', 'FIT_FUNC_XAXIS_LABELS'],
                             'literals': '0937033514159ccfd0f84c17e91c30c52e3a35b71df9d8e2986cda968f570da1',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': 'f6aa7ea1ba644b336e5e6a5d61b62120d268e8341cc4cd8c003c8cc8545f7ac1',
                             'key': ['assign', 'FIT_FUNC_NOISE_PARAM_LABEL'],
                             'literals': 'ce1517b9cd941ca94770d6bc6e069f067c09f36c9f4378bd00d4bec3e1bad08b',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '974f63de676ac1001d5a7360822d5dda9bd53132605ddfc11c6a7fbd8e6d735c',
                             'key': ['def', '_get_so_xaxis_label'],
                             'literals': 'd35918ead184a03ecb6bd2dd30acda7fef3d7e710dca333955d5379ee5f1bd08',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_get_so_xaxis_label'},
                            {'ast': '8138f0ee5c7dec1bac04a364492ecaaea7efc9881617cd15c590074925088292',
                             'key': ['def', '_get_noise_param_label'],
                             'literals': '0c026837a27ecb9caf5245af6a2115c11f59cdad81a71a1dd0e7dc1e8567c6e8',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_get_noise_param_label'},
                            {'ast': '7782ce57d06dfb6fb53af6816b4a73834d2d86cce433992380a97313f86e9b89',
                             'key': ['def', '_get_problem_goal'],
                             'literals': '5e3f564651cc228706c5664bf4bfc8483491713b0baac02b90e2adfa14434baf',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_get_problem_goal'},
                            {'ast': '869663987a7b1966f55c6fde5e5eade181fa35aadf349ac90a483b34a263c3c7',
                             'key': ['cb',
                                     '..schematic-graph.figure...schematic-legend.children..'],
                             'literals': '0e3b014f049ba53f64ea43d264b48f84027ce8619d275bd329e8a811aca34f44',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_schematic'},
                            {'ast': '1ab813c09eead8e29fa059e633bf4514e58befc5969441c7a16ed9274331bd0b',
                             'key': ['cb', 'experiment-description-display.children'],
                             'literals': '350110652918a2abef4376bbff03d6518232fc48a5bb471792f49ff0e95332ed',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_experiment_description'},
                            {'ast': '163d57af4be2608ba8e565245cdcc8c3ba827b70fe7469d94d926c7ce4663e48',
                             'key': ['cb', 'knapsack-info-display.children'],
                             'literals': 'b522a25146ececf5a5a524b3a39d96cc4c7ca788be198138d4764a96ca4b519a',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_knapsack_info'},
                            {'ast': 'ddc16fea1159c8731c1b821862c02b60aa09284d19740c101afa2deb9afabef4',
                             'key': ['cb', 'table1.data'],
                             'literals': '23ae3dab8b9a447fdf9d2ebf061acaf83af505251f520efce1c7dc6d5a0942b1',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table1_for_experiment'},
                            {'ast': 'a3def7b592b865e29404c6ba35bb1e0784eb310fdf4c004586dde57611bdaae0',
                             'key': ['cb', 'table1-selected-store.data'],
                             'literals': '8c0e77a8694491358038ff6baad885125e2ddaca687b62f57d54d436f725bc5d',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table1_store'},
                            {'ast': 'dab11d20a03f54edf066f0ea064be1b913c77063d7ce91aa3ec2ab6799218f2b',
                             'key': ['cb', 'table2-selected-store.data'],
                             'literals': 'd0eae5a51258a857c7a6d7c28857b223f76a9b3e0383a618893c0477784f94fa',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table2_store'},
                            {'ast': 'df2df7d29f62b3275a071a4bccc03bfc873abbf12445cac35c4ce7dafabdcb37',
                             'key': ['cb', 'data-problem-specific.data'],
                             'literals': 'c3b38d3553e756dc3bc1cc061e0a0673af81722dc847d56086a8ed28105e24e0',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'filter_table2'},
                            {'ast': '8905df5ea81b77ca8dd7e72ba06f7fe9ec906b4468a2725cf8f0c9e55c532143',
                             'key': ['cb', 'table2.data'],
                             'literals': 'd251a0cd16f9c8229a11f14d3d43e01740b8b9169e74277da1f02b0a09c6e9a0',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table2'},
                            {'ast': '237364dfd37ac97d556821a6730d43b83f09bba4cfcead81c3d06bdbeac651fe',
                             'key': ['cb',
                                     '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..'],
                             'literals': 'c591691b3c665d373600daf21c35260a75ad73f670f3a882490bc9d20b21b7d5',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table2'},
                            {'ast': '6b25e589a683a2ce18080b50941d3222d9dbc47c1d576d73900e46485bd31d9d',
                             'key': ['cb', 'table2-selected-output.children'],
                             'literals': '6ef6f2390d25f3b8a5d5e7bfc7a33dd972352adc4f553a5f31e1c91a71dc0c72',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table2_selected'},
                            {'ast': '67b3245699079b1e08b70ce41992c8869a87b3a3acc31bc2f745abb3d18f768a',
                             'key': ['assign', 'filter_columns'],
                             'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                             'module': 'dashboard/Dashboard.py',
                             'name': None},
                            {'ast': '23b8ec5ec7423f27a5e7e2e9b6515e934feed0eea26b8f7e88407d9b42816d78',
                             'key': ['def', '_filter_by_table2_selection'],
                             'literals': '1c33738e9703928c31bf0197e9786fb743f5def0ea1be9e5bc93234f872656db',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_filter_by_table2_selection'},
                            {'ast': '61a29fac7aec0852cafc06c3945077f876c9d83e9627898b2b15ee0b0a4b1beb',
                             'key': ['def', '_filter_penalty'],
                             'literals': '676fcec809521dbd186bb41691fa8cba94224673cd56cec96ba8e321a88194ac',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_filter_penalty'},
                            {'ast': '8c8f8d1aa016733e035a174f1f0fa5ac9d29c6def54132c7c172284d3b129beb',
                             'key': ['cb', 'penalty-filter-dropdown.options'],
                             'literals': '99d97ede88ca6c0a103be0a4935a21ad36316767b9b309b2a435da78edca924e',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_penalty_filter_options'},
                            {'ast': '294e125c2a15ec0f3542ae2c253784acbeac1586660a3f1bcc73adade50c68d5',
                             'key': ['cb', 'penalty-filter-dropdown.value'],
                             'literals': 'a628237ddfcf55403967b83a24f0e64309508617587e0e091dbf8a0e540510fd',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'reset_penalty_filter_on_problem_change'},
                            {'ast': 'cb4f7645a7aa2362578461fa39420ff1837130404e39b7fefeeea9019fe91385',
                             'key': ['cb', 'plot_2d_data.data'],
                             'literals': 'f5a8360cf0142d8e2f58bebc2957497cbc67159c1e9eedf9770dd8d95ade57a5',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_filtered_view'},
                            {'ast': '59cf9b3bc8ad75a2deb32ca733d18abd1e000007da2f99d36404c3448c5c5f34',
                             'key': ['def', '_cap_noise'],
                             'literals': 'c78e6b95fdc730a5b7953e7b1f3b5769d1c9a06162f5f002865fd5100bd7752e',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_cap_noise'},
                            {'ast': 'e70d75bc1afcb0d7c824c7f043be013e52a9fb4d229e45ab8c282bf8c44d0d7a',
                             'key': ['def', '_hide_series'],
                             'literals': '9c26bb0b95dcc26da2dc15aec2ec3740840bb6cbaa70ddb7a370431a8d142198',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_hide_series'},
                            {'ast': '15bdd0ba1415dcdbe74d1ec19063fd97b51b83460a9aebf4026a5d5d405f7b56',
                             'key': ['def', '_resolve_evals_column'],
                             'literals': '6ec7531cce20a76ef6a40f146c5d655dc326c29af137715fa51a373d3ecf502c',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_resolve_evals_column'},
                            {'ast': '1e1b98644e30a6338c1a2259f3ca5cd993d12c58191ce5d48b45ecaa1c1a2172',
                             'key': ['def', '_format_scientific'],
                             'literals': 'bd82b0149de9fc60b90f1e9d819c41110fb03368c26ba21e549654fbb23f3d23',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_format_scientific'},
                            {'ast': 'e7c35cd2186c7adcee6b7b8baa4aa97dc2133408c901e489521aa07c2860bbd3',
                             'key': ['def', '_format_median_std'],
                             'literals': '0c2cc6a794799c1b1cb9159945af57291d1060b0898bea2c78afda20276ab37e',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_format_median_std'},
                            {'ast': '0e0821be9500a2c1e1c87aea3025c154dc68d21fb9511f9b62223268aa63a02f',
                             'key': ['def', '_resolve_fit_column'],
                             'literals': '0294ce090c07f7a041828487b83e4f9536b2530824dadb41962260130c98c1e3',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_resolve_fit_column'},
                            {'ast': '134c3a0039491079c9545365ed18459f88b965694ec22835fd657f4e8ca8e589',
                             'key': ['cb', 'hide-series-dropdown.options'],
                             'literals': '380ade669b2031677644ac25cc03bfd506566f3ad866081fee5a9fdcf8a49555',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_hide_series_options'},
                            {'ast': '33cfc24420830ae07627ca0761ce055b0363defa91b06150306fdded53099690',
                             'key': ['cb',
                                     '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..'],
                             'literals': '24fd5cbee18b0697ae6bc6bc33cbbf437090948e4cd663a7a8600487271f6345',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_advanced_misjudgement_algo_options'},
                            {'ast': 'b67ac7703911faae4e67dd03674eed92b88aad5278a578f5c15f522e8cb14ad2',
                             'key': ['cb', 'plot_2d_data_table.data'],
                             'literals': '99e07f820cca60d1ff9ffa2781aa29296a1c955fbde32c6d2898d28758e0468a',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data'},
                            {'ast': 'ff0fff920a82c6b39b135ba16bd1e0b2807e6d9a3e71759c6180e2e209807e83',
                             'key': ['cb', '2DLinePlot.figure'],
                             'literals': '19431fe5f2fa249e07b1c70b9f0b56ffa550e5eee4b702e47d9f466c3b893177',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data'},
                            {'ast': 'f320dd61f7c70b847151df680edaa69d63347ede858af2aebbbe54e1efcd7771',
                             'key': ['cb', '2DBoxPlot.figure'],
                             'literals': 'f801d78d14043150868f9f281477905b807206eaa445af647ce38a7623477fff',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data'},
                            {'ast': '6421778fdee8cfed0e9e464bf25fa0ca4439ea892df790397e2439a971ebc06c',
                             'key': ['cb', '2DLinePlotMO.figure'],
                             'literals': '6f2b0462509663d0533b5a84a8895f3d02f7704bb617a879105735be71563232',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data_mo_line'},
                            {'ast': '0ccd6a107262027245a71afc236c47dfb8b71923728fa868f96a47dcb0efa4c7',
                             'key': ['cb', '2DBoxPlotMO.figure'],
                             'literals': '5cec1a92bc632029cb9a8437918566fd9818cf7b3eee5734838c588346ab38d2',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_stored_data_mo_box'},
                            {'ast': '223454b6a4b5fdec35f60d3e83c4331e7284cfd489426b6aeedee1cac60be41e',
                             'key': ['cb', '2DLinePlotEvalsSO.figure'],
                             'literals': 'c9c904491d1ea9cb678d313eb3a583dd8b23876489efb0d2a2178e02efd04f7a',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_line_evals_so'},
                            {'ast': '6e1bd5a6f4afc19ea05eaa5349f32a16ac6c3a277469be65ca0a25a093662672',
                             'key': ['cb', '2DBoxPlotEvalsSO.figure'],
                             'literals': '23745611122f4e23cfeccc28566a7383e377d773885690c1b52514d71d923f24',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_box_evals_so'},
                            {'ast': '64c160326df26264a0bb7750638c50bd9b9d7bd66b51429391b7cf1a726cfa56',
                             'key': ['cb', '2DBoxPlotPenaltySO.figure'],
                             'literals': 'a7032ad9cdb29ff56de57f86366e79fdbaf06be366c903e83e3cd3a4e71ab988',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_box_penalty_so'},
                            {'ast': '6f5f7297aa260f474d05146b9b9fb3b5f531912012ee25043776048eb2a95d6f',
                             'key': ['cb', '2DBoxPlotMisjudgementsSO.figure'],
                             'literals': 'efe1d13479cbc103bfbcfd55ecba4dde3abd878628100a47ab32202070e62428',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_box_misjudgements_so'},
                            {'ast': 'c7985951f33fb5c0f12b6cf261c333936dac664725ab85afa0b25ac644da74b1',
                             'key': ['cb', '2DBoxAdvancedMisjudgementsSO.figure'],
                             'literals': 'dcc02d643887c936b6c06e68da40b6c4ee11afee66cc5751693a0340971b76e6',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'display_box_advanced_misjudgements_so'},
                            {'ast': 'f1ae187a02b37dd6558d583de2b6c461bae6db5be5147a4fdcf8b96ba50a0f8b',
                             'key': ['cb', 'misjudgements-summary-table.children'],
                             'literals': 'c59734fd17bb04d7057c60a314a7ca8b84612ca4565b8d376c089686ca3b8a86',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_misjudgements_summary_table'},
                            {'ast': '08179e57346514b98f524449ef7c900823757468e2b3705a83c795f86c9c68a8',
                             'key': ['cb', 'performance-summary-table.children'],
                             'literals': 'f632c4402670a99df36cd880f239ae59e6e7a776438007a3471a8f802f165acf',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_performance_summary_table'},
                            {'ast': 'df9d5fce5d0395bc433e11f6dbb2065a50b55672d53bd2e851d61ce95be40ae5',
                             'key': ['cb', 'mann-whitney-table.children'],
                             'literals': '3380bfd4df668613711496b082798d07727450a36508102f2344475f213767a3',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_mann_whitney_table'},
                            {'ast': 'b50c40d77ac909f7783cf5b5be4619981553d5dcc574a3802dff74bb532447cb',
                             'key': ['cb', 'evals-summary-table.children'],
                             'literals': '77d3ce03f655a9260ed6ccbb24a99020011fad0fc164044c729d372865b44e59',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_evals_summary_table'},
                            {'ast': '27d2f772d4ca31e40bfbea7868778fcdc324b449f1d5672d17b715d044b57f5f',
                             'key': ['cb', 'evals-mann-whitney-table.children'],
                             'literals': '77854eb3232e7d3808ceb6a4dd1c6cf72635e95a3d1cea30b0bfe3d4383cfecb',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_evals_mann_whitney_table'},
                            {'ast': '479673a66ef7511fddea028bedf5753ec822c02e28c33a9e3e41085f2893ef8a',
                             'key': ['cb', '2DPlotTabContent.children'],
                             'literals': '7e996af29e6dd93a6cdca04c45a0736a41440cd6a8c6802428972d1a11784571',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'render_content_2DPlot_tab'},
                            {'ast': '2c21948bfbbaf154fcae6d77ebe14256cddc4b4224c86023a29670f5e62c4297',
                             'key': ['cb', 'lon-table-selected-pid-store.data'],
                             'literals': '01a7ee181a3b25ad2a3586c9dc94a01055d01aa1b9f83770c18df477cf8ce1b9',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_lon_table_selected_pid'},
                            {'ast': '0d81d258980d4881aefeef07bff97f24cde23de0632c04c12db703caf243fe87',
                             'key': ['cb', 'LON_data.data'],
                             'literals': '6fb69bf7487dff26ecab22eff8d3804674436f1899944081d67b102b26d8d469',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_filtered_view'},
                            {'ast': '4374850ceaef8b936826fe22bf9232f9c0ed7fcb191d4de6e1e12efee734ae92',
                             'key': ['cb', 'STN_data.data'],
                             'literals': 'a4e1ecfec49831194e6d4f8cc40d23a063262a6a0e5cb6f9a9565103b5a7cea5',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_filtered_view'},
                            {'ast': 'b3a08a5eaa1082b8bb2a68a50fd76f90509759455154938abb5101a3f6a31f4b',
                             'key': ['cb',
                                     '..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..'],
                             'literals': '837e8d5b88f3efde50184fe8ed40b24368a05a93b22c553d8e31f6579c3b5021',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'process_STN_data'},
                            {'ast': '17bf6317af520df4d9fdef41107609f2353e2d8189c2a6732714f825addf4741',
                             'key': ['cb', 'run-print-info.style'],
                             'literals': '6bfa310fd1a4e7fff91cd0235d7c0db86b65e9165127b0054bd52eb70131e66b',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'toggle_run_print_info'},
                            {'ast': '2b7ee9b421dcd601e072e58aef6ea6b27775bd53e7babab98a67e88247ba0716',
                             'key': ['cb', 'print_STN_series_labels.children'],
                             'literals': '48a4b42e43cd07e2cbe8613e36cb3f150da005b7424a5fd8f4be09f36a333c27',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_table2_selected'},
                            {'ast': 'c20086fc69c55ca534d1365788060f3654ab875db5785430ff0e5a5ea5856ad8',
                             'key': ['cb', 'axis-values.data'],
                             'literals': 'a89565368bc3c07d609e2e3c23f34e6ea7418f08816f87b0a4abbaf161010609',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'clean_axis_values'},
                            {'ast': 'a7460f7aafb20e55505edc6a95dce0e4c6e116bd4c86f62cc63b92a5c4eb94f9',
                             'key': ['def', '_build_stn_stats_table'],
                             'literals': '6d22dda24431cd54cfad1df8c16f1e7f011fe5faebce036d88cb33e777687830',
                             'module': 'dashboard/Dashboard.py',
                             'name': '_build_stn_stats_table'},
                            {'ast': '11f587de3e0a87d1d3213b958b221b6425c8d57b2c36ab8fa7aea40609c7851a',
                             'key': ['cb',
                                     '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..'],
                             'literals': 'be02d83664a644a710d01376409d09088dda76cc66c428247c7429a1c6a67743',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'handle_print_mode'},
                            {'ast': 'f6f6fcddeef5c8820e54b523c97e6b6e93a93af8ea4c36f7f17014f2784ac5d5',
                             'key': ['cb',
                                     '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..'],
                             'literals': 'f51280651ead6cd2aabe12d5dfd46f47adc2e919410cbc538c757a78bf83328c',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'update_plot'},
                            {'ast': '55fef27776e6f3a0fde076681b9a55e21ca8202edd6252e9e64856eb7eaf0052',
                             'key': ['cb', 'plotParetoFront.figure'],
                             'literals': '3de33cb4cedf04ac528635238127681a8f5ef39ed76a880f88949a73462a3429',
                             'module': 'dashboard/Dashboard.py',
                             'name': 'updateParetoPlot'},
                            {'ast': '8fc2ecd3bdedcb87d2fdda4979382034bcd3513ec008491ee94ddb6905bec02a',
                             'key': ['main'],
                             'literals': 'af3f4b5db7c7166abef951cf198d6a6a67c7f32ae843ef7e4812e0a2b1ec9b70',
                             'module': 'dashboard/Dashboard.py',
                             'name': None}],
 'dashboard/DashboardHelpers.py': [{'ast': 'f110e357e9c54f0a89a8b1575d8d76f074d31a0853298477c3636afee883ee30',
                                    'key': ['doc'],
                                    'literals': '4833ead9195e74cb4cb1b0e1f9ed0744f02e9eff69146fe878b7c1600f05c230',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': None},
                                   {'ast': '2e4bdd609cfa37c234d1b1397284cb8dad2fa21cce21f483e00159e86878655a',
                                    'key': ['def', 'convert_to_rgba'],
                                    'literals': 'bda3d79c0b9671843227281ced753f51fd6db2f8286f09954e2feb8c1749614c',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'convert_to_rgba'},
                                   {'ast': 'c118d07d2c3080c86ef01e19a4013fd9440adf25de2b01fc859add0254ea5e11',
                                    'key': ['def', 'fitness_to_color'],
                                    'literals': '49f8db9d99799f75ceb7adf2a973065ee49d26f61ecdd9852f911fe5792df4f3',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'fitness_to_color'},
                                   {'ast': 'f7551569ea5c14a0048252e0a19c1db48486616f53d09b202ea62b8ebf984470',
                                    'key': ['def', 'select_top_runs_by_fitness'],
                                    'literals': 'db27e41711dd4e59a28bbece86b1f279e48a154a7712a4dc453ccedead980be1',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'select_top_runs_by_fitness'},
                                   {'ast': 'd00157fbcb3ca236874aa02cd388d1107037bb8743cead7d101a231df654664c',
                                    'key': ['def', 'get_mean_run'],
                                    'literals': '85f9c634edf74aed82396bf0eaacc7212f58c77c4ad707c75455a98a6609372b',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'get_mean_run'},
                                   {'ast': 'd402e0019a63781d6ef03f018ab25e84bb3cb7f108458f1274384dcabd8c3fce',
                                    'key': ['def', 'get_median_run'],
                                    'literals': '892291e30df8be44e5f3475a2a3638027ac4491e33baef29f9c2dde7d531d406',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'get_median_run'},
                                   {'ast': 'ed857acdcd72bd3d61285b1852335e027f0a67620f7763163f0b1f81be5feacf',
                                    'key': ['def', 'determine_optimisation_goal'],
                                    'literals': 'c96eebe67b41d91b76413dae9b6998cfdaa1d389dc1a05cd6565dc4a03eeb821',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'determine_optimisation_goal'},
                                   {'ast': '4a78f9bfcc0e5f0a54fbf8c1288cbd3fb53d419058f8db6e8fc678049f81ee3c',
                                    'key': ['def', 'filter_negative_LO'],
                                    'literals': 'e4e2c25b1e564e4f37634d686ada6869fa94e24db43ba5fa041eb0e8fc64aec5',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'filter_negative_LO'},
                                   {'ast': 'cf3ae769a9afc9602014c11d5555f521b4906933f2adfe0d9494137fd69ca9c9',
                                    'key': ['def', 'convert_to_split_edges_format'],
                                    'literals': 'a64b556c84abd0a6336cadfea37d20997594e8830fbd2bed753916c123eaf04f',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'convert_to_split_edges_format'},
                                   {'ast': 'c2aa9bc982f76b6e0b4db751a90e892f1a37b4163ec5967970c293b75e5c081c',
                                    'key': ['def', 'convert_to_single_edges_format'],
                                    'literals': 'ad8ad2c0430bcfd806f8c78a2411afa7aa2fe0ee4ba559cec26130271efbfcb9',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'convert_to_single_edges_format'},
                                   {'ast': 'c5ca48a389db20b6275e538dbfacb26d5d6675a1a51f6f2008fcd53e63cac1cc',
                                    'key': ['def', 'filter_local_optima'],
                                    'literals': '8e588def216b4fb93757c0406a8060b7560b369f1b1166bee74e34f1f61a258d',
                                    'module': 'dashboard/DashboardHelpers.py',
                                    'name': 'filter_local_optima'}],
 'dataio/__init__.py': [{'ast': '28ac9204dec5f8be3b1b73a715576f5fba8f56f7327238b68d35b333cbc71e25',
                         'key': ['doc'],
                         'literals': '543b1ba601eda069b12a939cd52768fb4e733b2e41bb8b32167c0002d0637fac',
                         'module': 'dataio/__init__.py',
                         'name': None},
                        {'ast': 'c94575a78680da57e93718b550abc59ffb76c7559ad9d6ba8f9c3fc37ea54920',
                         'key': ['def', 'DashboardData'],
                         'literals': '447abf5940f4cc55239a1c9d3839bf18c014bf750b80d240d9bebbca2e4ea853',
                         'module': 'dataio/__init__.py',
                         'name': 'DashboardData'},
                        {'ast': 'de41b84c01033149fcc0f28604cf98ebaf4ef59d7360aa47234dae8f83fcfe5c',
                         'key': ['assign', '__all__'],
                         'literals': '7bad43ced07efd114ab65f908c54aee93153149793a14b21c777b08e6cfe7099',
                         'module': 'dataio/__init__.py',
                         'name': None}],
 'dataio/column_config.py': [{'ast': 'd706caa5544c2236babb0e473b8e84ed08e0c68ffe9a3c99204530c83de7ec5e',
                              'key': ['doc'],
                              'literals': 'a7cb200a98fd0ea3a0a7ebd08d5aff712393e21aa058fe177009fadb35207af9',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': 'e589ad6980bd47e3408a46d711b38c60d3d663efaa18bb4fdd78013cda6ce190',
                              'key': ['assign', 'LON_HIDDEN_COLUMNS'],
                              'literals': '4e78d7298cbd046379e17405a7073c3bc67def9069ff73e04243d14216d248b3',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': 'ad9bf1502bac4d3fd4fd74630355d9344a9cccea6fc469280cb11c433c98a129',
                              'key': ['assign', 'DISPLAY2_HIDDEN_COLUMNS'],
                              'literals': '29b11053a12a47a1ad36e6d324bf7df086f10972d859d28a698b9b92bb692c04',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': '0c60673ad473f5cf1acf4b0d475b12df05c8b90a8c8c964e12a7955dd344b07b',
                              'key': ['assign', 'DISPLAY1_COLUMNS'],
                              'literals': '065fba40db10d1cbe4eb980e58662af40914eacbb60c0b4760e11fe7ae3d0a0a',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': 'cf2c3a5e6b975ddd49880218909af29dbfb47c45e141273018d4867ebabbbcc8',
                              'key': ['assign', 'LIST_COLUMNS'],
                              'literals': '6d9695426f66f8afd8fef7fd8e75e7a8dedeaabd2b1cca8ce2864ee9d71dc33c',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': 'c39561eb2b8528a93fe9dec5264a05b9cbe09631d0df5d7f430a3c0966b110f8',
                              'key': ['assign', 'DISPLAY2_DROP_COLUMNS'],
                              'literals': 'fc0fbf5601fdbb31763aafd043516da48930cfcd3452c6f1a80addc5e32697af',
                              'module': 'dataio/column_config.py',
                              'name': None},
                             {'ast': '5aa33b52bfd3bc345774aebc072c982bcac54badc37c347eaad66d351fc20699',
                              'key': ['assign', 'DISPLAY2_DEDUP_KEYS'],
                              'literals': 'f40715694f4bc1806e1c5530ff9f0186208bbe1dcb56195a5a011604d5846530',
                              'module': 'dataio/column_config.py',
                              'name': None}],
 'dataio/transformers.py': [{'ast': 'c3154c2c7579034c0088f7a83b06971f70b2e32b64d76c0a7b96c73b985eba2a',
                             'key': ['doc'],
                             'literals': 'a49a07794591923fd0df32c5428c1b2fa70006a86cd2a346842f88b9d67f5a0b',
                             'module': 'dataio/transformers.py',
                             'name': None},
                            {'ast': '8e343ad81a2e4116cc54fe409dee2657ea40bfd86fc7953d3c19e305007fa193',
                             'key': ['def', '_compute_n_misjudgements'],
                             'literals': '0be443eabc002bfcd178d5f903e11c1f101a6fca894cf626c81087413928bdc0',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_n_misjudgements'},
                            {'ast': '26b6dbd0e4f38a23a642968a1877050b151fb02f5f9eb1a5b7f2351a7cb0dd55',
                             'key': ['def', 'increasing_noise_step_indices'],
                             'literals': '1fbc61e6525c203f7c9f86f4d682eeec1d4f2a8d91a1be99636cb320e5653a12',
                             'module': 'dataio/transformers.py',
                             'name': 'increasing_noise_step_indices'},
                            {'ast': '49dbc10692f2e1c0d57768151637c1bd57072c058f4c41159379c745eff27a17',
                             'key': ['def', 'comparison_misjudgement_step_indices'],
                             'literals': 'acfb92df09209dd9449047ec7f3de6c0550deeab5bb71d94039075d576c77c0a',
                             'module': 'dataio/transformers.py',
                             'name': 'comparison_misjudgement_step_indices'},
                            {'ast': '5079976cc5cb40cedbc53ef8fd53e5adf708c44f8ff676acbcb798ead6c49cae',
                             'key': ['def', 'constraint_misjudgement_step_indices'],
                             'literals': 'dd8b86ea92bf34bec1fb5219446799ae7b6525b7ecf37f61d497232fc67daef3',
                             'module': 'dataio/transformers.py',
                             'name': 'constraint_misjudgement_step_indices'},
                            {'ast': 'a512731cbf8e80212bfa4ad26b1e257d39358f8e424b4d47411ebbf6a7802138',
                             'key': ['def', '_compute_n_increasing_noise'],
                             'literals': '41cb3d28749be116c7a9cd534d82777bef8de8efb91a1fa03a3fef036ec41990',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_n_increasing_noise'},
                            {'ast': '13401a4eee4eaf5f237cd0ab05c3a162b0b02056b12d985d0260dc54076fe08d',
                             'key': ['def', '_compute_n_comparison_misjudgements'],
                             'literals': '60e2877381688371a5ded0717aa6678c8ced359359bacf8f119985eeabe99390',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_n_comparison_misjudgements'},
                            {'ast': '2637b0872616ba9d298d6beec93bc7bea005d8ac505c11e07f709d2e5c5b8a70',
                             'key': ['def', '_compute_n_constraint_misjudgements'],
                             'literals': 'b881d0c72fff05b8fe6a267de026535ee7e6ee0bd18591625aa6515d2495e088',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_n_constraint_misjudgements'},
                            {'ast': '9e89afef39d1758adb6596d4e3ff081e0b4873be2287ca2bf0f82170ddb72bd8',
                             'key': ['def', '_evals_to_visit'],
                             'literals': 'b087b446391a9e1c5a8d4fda77381fec1cf710510de219d9b7ec9925851cba85',
                             'module': 'dataio/transformers.py',
                             'name': '_evals_to_visit'},
                            {'ast': 'af4f9ad43f96383e44cf0157aca50bff3909c83a74b1c560ee3166c2eff8e829',
                             'key': ['def', '_compute_evals_to_best'],
                             'literals': '379b0465f7e1adb406674e07b96c569bb3376c553a4868393d986a15864a870c',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_evals_to_best'},
                            {'ast': 'd109856fb480522c24f0e1ba9510dffc197c7dcfd2de4bd5dd2e5e5e86fb9692',
                             'key': ['def', '_compute_evals_to_final'],
                             'literals': 'bdccc2b79d7a15178359b9fcf86d223acf4004e9313b83ba4dde3ad7317eab2e',
                             'module': 'dataio/transformers.py',
                             'name': '_compute_evals_to_final'},
                            {'ast': '91ffa21fef9c5455e600faab2c53cdd56732e93c455c71a05ae91a0601ded068',
                             'key': ['def', 'create_df_no_lists'],
                             'literals': 'f96e0e39066634cb95bccc976c40bb41177938f750eb67dd2227ccd6ee70e42f',
                             'module': 'dataio/transformers.py',
                             'name': 'create_df_no_lists'},
                            {'ast': 'a28291583bd576cf64608d98d86b2064d52320bf9bec197235099af1e4c852c4',
                             'key': ['def', 'create_display1_df'],
                             'literals': '760959a879e24b3a8e6c8a15bd79d5381ddff99b921da81846d0ffdf88e12746',
                             'module': 'dataio/transformers.py',
                             'name': 'create_display1_df'},
                            {'ast': 'e6df04a70682f8837a0ec4936dda966ff5013c78d0fe99df9fa8efdffb993be1',
                             'key': ['def', 'create_display2_df'],
                             'literals': '0d47269c4815f6a5cb004053a012639b78fe68385e236e6911fa7cec3c8bed66',
                             'module': 'dataio/transformers.py',
                             'name': 'create_display2_df'},
                            {'ast': 'c049f195bb6259823c57364df5ad80d78d6b5ebdeb3cf082541e2c95b8d2c68e',
                             'key': ['def', 'get_lon_display_columns'],
                             'literals': '1be1792e0bb50e09521f8ef12632f72a4b346f8b2642e2e040a3918ca30fb0be',
                             'module': 'dataio/transformers.py',
                             'name': 'get_lon_display_columns'}]}

CALLBACK_MODULES = {'..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': 'dashboard/callbacks/graph_data.py',
 '..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..': 'dashboard/callbacks/performance.py',
 '..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..': 'dashboard/callbacks/visualization.py',
 '..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': 'dashboard/callbacks/selection.py',
 '..schematic-graph.figure...schematic-legend.children..': 'dashboard/callbacks/schematic.py',
 '..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': 'dashboard/callbacks/visualization.py',
 '2DBoxAdvancedMisjudgementsSO.figure': 'dashboard/callbacks/performance.py',
 '2DBoxPlot.figure': 'dashboard/callbacks/performance.py',
 '2DBoxPlotEvalsSO.figure': 'dashboard/callbacks/performance.py',
 '2DBoxPlotMO.figure': 'dashboard/callbacks/performance.py',
 '2DBoxPlotMisjudgementsSO.figure': 'dashboard/callbacks/performance.py',
 '2DBoxPlotPenaltySO.figure': 'dashboard/callbacks/performance.py',
 '2DLinePlot.figure': 'dashboard/callbacks/performance.py',
 '2DLinePlotEvalsSO.figure': 'dashboard/callbacks/performance.py',
 '2DLinePlotMO.figure': 'dashboard/callbacks/performance.py',
 '2DPlotTabContent.children': 'dashboard/callbacks/performance.py',
 'LON_data.data': 'dashboard/callbacks/graph_data.py',
 'STN_data.data': 'dashboard/callbacks/graph_data.py',
 'axis-values.data': 'dashboard/callbacks/visualization.py',
 'data-problem-specific.data': 'dashboard/callbacks/selection.py',
 'evals-mann-whitney-table.children': 'dashboard/callbacks/performance.py',
 'evals-summary-table.children': 'dashboard/callbacks/performance.py',
 'experiment-description-display.children': 'dashboard/callbacks/selection.py',
 'hide-series-dropdown.options': 'dashboard/callbacks/performance.py',
 'knapsack-info-display.children': 'dashboard/callbacks/selection.py',
 'lon-table-selected-pid-store.data': 'dashboard/callbacks/graph_data.py',
 'mann-whitney-table.children': 'dashboard/callbacks/performance.py',
 'misjudgements-summary-table.children': 'dashboard/callbacks/performance.py',
 'penalty-filter-dropdown.options': 'dashboard/callbacks/performance.py',
 'penalty-filter-dropdown.value': 'dashboard/callbacks/performance.py',
 'performance-summary-table.children': 'dashboard/callbacks/performance.py',
 'plotParetoFront.figure': 'dashboard/callbacks/pareto.py',
 'plot_2d_data.data': 'dashboard/callbacks/performance.py',
 'plot_2d_data_table.data': 'dashboard/callbacks/performance.py',
 'print_STN_series_labels.children': 'dashboard/callbacks/visualization.py',
 'run-print-info.style': 'dashboard/callbacks/visualization.py',
 'table1-selected-store.data': 'dashboard/callbacks/selection.py',
 'table1.data': 'dashboard/callbacks/selection.py',
 'table2-selected-output.children': 'dashboard/callbacks/selection.py',
 'table2-selected-store.data': 'dashboard/callbacks/selection.py',
 'table2.data': 'dashboard/callbacks/selection.py'}

STATEMENT_DESTINATIONS = {'app/app.py': {'assign|app': 'mlflow_app/app.py',
                'assign|app.layout': 'mlflow_app/app.py',
                'main|': 'mlflow_app/app.py'},
 'app/pages/mlflow_browser.py': {'assign|EXP_COLS': 'mlflow_app/pages/mlflow_browser.py',
                                 'assign|RUN_COLS_DEFAULT': 'mlflow_app/pages/mlflow_browser.py',
                                 'assign|layout': 'mlflow_app/pages/mlflow_browser.py',
                                 'def|remember_runs': 'mlflow_app/pages/mlflow_browser.py',
                                 'def|render_experiments': 'mlflow_app/pages/mlflow_browser.py',
                                 'def|render_runs_table': 'mlflow_app/pages/mlflow_browser.py',
                                 "expr|dash.register_page(__name__, path='/experiments', name='Experiments')": 'mlflow_app/pages/mlflow_browser.py'},
 'dashboard/Dashboard.py': {'assign|FIT_FUNC_NOISE_PARAM_LABEL': 'dashboard/helpers.py',
                            'assign|FIT_FUNC_XAXIS_LABELS': 'dashboard/helpers.py',
                            'assign|LON_display_columns': 'dashboard/data.py',
                            'assign|LON_hidden_cols': 'dashboard/data.py',
                            'assign|app': 'dashboard/instance.py',
                            'assign|app.layout': 'dashboard/app.py',
                            'assign|data': 'dashboard/data.py',
                            'assign|df': 'dashboard/data.py',
                            'assign|df_LONs': 'dashboard/data.py',
                            'assign|df_no_lists': 'dashboard/data.py',
                            'assign|display1_df': 'dashboard/data.py',
                            'assign|display2_df': 'dashboard/data.py',
                            'assign|display2_hidden_cols': 'dashboard/data.py',
                            'assign|experiment_descriptions': 'dashboard/data.py',
                            'assign|experiment_names': 'dashboard/data.py',
                            'assign|filter_columns': 'dashboard/callbacks/performance.py',
                            'assign|tab_selected_style': 'dashboard/callbacks/performance.py',
                            'assign|tab_style': 'dashboard/callbacks/performance.py',
                            'cb|..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data...STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..': 'dashboard/callbacks/graph_data.py',
                            'cb|..advanced-misjudgement-algo-dropdown.options...advanced-misjudgement-algo-dropdown.value..': 'dashboard/callbacks/performance.py',
                            'cb|..annotation-options.value...axes-text-scale.value...annotation-text-scale.value..': 'dashboard/callbacks/visualization.py',
                            'cb|..optimum.data...PID.data...opt_goal.data...fit_func_store.data..': 'dashboard/callbacks/selection.py',
                            'cb|..schematic-graph.figure...schematic-legend.children..': 'dashboard/callbacks/schematic.py',
                            'cb|..trajectory-plot.figure...run-print-info.children...stn-stats-table.children...lon-stats-table.children...lon-feas-error-scatter.figure...lon-selected-correlation.children...lon-feas-error-correlations.children..': 'dashboard/callbacks/visualization.py',
                            'cb|2DBoxAdvancedMisjudgementsSO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DBoxPlot.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DBoxPlotEvalsSO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DBoxPlotMO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DBoxPlotMisjudgementsSO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DBoxPlotPenaltySO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DLinePlot.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DLinePlotEvalsSO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DLinePlotMO.figure': 'dashboard/callbacks/performance.py',
                            'cb|2DPlotTabContent.children': 'dashboard/callbacks/performance.py',
                            'cb|LON_data.data': 'dashboard/callbacks/graph_data.py',
                            'cb|STN_data.data': 'dashboard/callbacks/graph_data.py',
                            'cb|axis-values.data': 'dashboard/callbacks/visualization.py',
                            'cb|data-problem-specific.data': 'dashboard/callbacks/selection.py',
                            'cb|evals-mann-whitney-table.children': 'dashboard/callbacks/performance.py',
                            'cb|evals-summary-table.children': 'dashboard/callbacks/performance.py',
                            'cb|experiment-description-display.children': 'dashboard/callbacks/selection.py',
                            'cb|hide-series-dropdown.options': 'dashboard/callbacks/performance.py',
                            'cb|knapsack-info-display.children': 'dashboard/callbacks/selection.py',
                            'cb|lon-table-selected-pid-store.data': 'dashboard/callbacks/graph_data.py',
                            'cb|mann-whitney-table.children': 'dashboard/callbacks/performance.py',
                            'cb|misjudgements-summary-table.children': 'dashboard/callbacks/performance.py',
                            'cb|penalty-filter-dropdown.options': 'dashboard/callbacks/performance.py',
                            'cb|penalty-filter-dropdown.value': 'dashboard/callbacks/performance.py',
                            'cb|performance-summary-table.children': 'dashboard/callbacks/performance.py',
                            'cb|plotParetoFront.figure': 'dashboard/callbacks/pareto.py',
                            'cb|plot_2d_data.data': 'dashboard/callbacks/performance.py',
                            'cb|plot_2d_data_table.data': 'dashboard/callbacks/performance.py',
                            'cb|print_STN_series_labels.children': 'dashboard/callbacks/visualization.py',
                            'cb|run-print-info.style': 'dashboard/callbacks/visualization.py',
                            'cb|table1-selected-store.data': 'dashboard/callbacks/selection.py',
                            'cb|table1.data': 'dashboard/callbacks/selection.py',
                            'cb|table2-selected-output.children': 'dashboard/callbacks/selection.py',
                            'cb|table2-selected-store.data': 'dashboard/callbacks/selection.py',
                            'cb|table2.data': 'dashboard/callbacks/selection.py',
                            'def|_add_guide_nodes': 'dashboard/callbacks/visualization.py',
                            'def|_build_stn_stats_table': 'dashboard/callbacks/visualization.py',
                            'def|_cap_noise': 'dashboard/callbacks/performance.py',
                            'def|_filter_by_experiment': 'dashboard/callbacks/selection.py',
                            'def|_filter_by_table2_selection': 'dashboard/callbacks/performance.py',
                            'def|_filter_penalty': 'dashboard/helpers.py',
                            'def|_format_median_std': 'dashboard/callbacks/performance.py',
                            'def|_format_scientific': 'dashboard/callbacks/performance.py',
                            'def|_get_noise_param_label': 'dashboard/helpers.py',
                            'def|_get_problem_goal': 'dashboard/helpers.py',
                            'def|_get_so_xaxis_label': 'dashboard/helpers.py',
                            'def|_hide_series': 'dashboard/callbacks/performance.py',
                            'def|_resolve_evals_column': 'dashboard/callbacks/performance.py',
                            'def|_resolve_fit_column': 'dashboard/callbacks/performance.py',
                            "if|'experiment_description' in df.columns and 'experiment_name' in df.columns": 'dashboard/data.py',
                            'main|': 'dashboard/app.py'},
 'dashboard/DashboardHelpers.py': {'def|convert_to_rgba': 'dashboard/helpers.py',
                                   'def|convert_to_single_edges_format': 'dashboard/helpers.py',
                                   'def|convert_to_split_edges_format': 'dashboard/helpers.py',
                                   'def|determine_optimisation_goal': 'dashboard/helpers.py',
                                   'def|filter_local_optima': 'dashboard/helpers.py',
                                   'def|filter_negative_LO': 'dashboard/helpers.py',
                                   'def|fitness_to_color': 'dashboard/helpers.py',
                                   'def|get_mean_run': 'dashboard/helpers.py',
                                   'def|get_median_run': 'dashboard/helpers.py',
                                   'def|select_top_runs_by_fitness': 'dashboard/helpers.py',
                                   'doc|': 'dashboard/helpers.py'},
 'dataio/__init__.py': {'assign|__all__': None,
                        'def|DashboardData': 'dashboard/data.py',
                        'doc|': 'dashboard/data.py'},
 'dataio/column_config.py': {'assign|DISPLAY1_COLUMNS': 'dashboard/columns.py',
                             'assign|DISPLAY2_DEDUP_KEYS': 'dashboard/columns.py',
                             'assign|DISPLAY2_DROP_COLUMNS': 'dashboard/columns.py',
                             'assign|DISPLAY2_HIDDEN_COLUMNS': 'dashboard/columns.py',
                             'assign|LIST_COLUMNS': 'dashboard/columns.py',
                             'assign|LON_HIDDEN_COLUMNS': 'dashboard/columns.py',
                             'doc|': 'dashboard/columns.py'},
 'dataio/transformers.py': {'def|_compute_evals_to_best': 'dashboard/tables.py',
                            'def|_compute_evals_to_final': 'dashboard/tables.py',
                            'def|_compute_n_comparison_misjudgements': 'analysis/misjudgements.py',
                            'def|_compute_n_constraint_misjudgements': 'analysis/misjudgements.py',
                            'def|_compute_n_increasing_noise': 'analysis/misjudgements.py',
                            'def|_compute_n_misjudgements': 'analysis/misjudgements.py',
                            'def|_evals_to_visit': 'dashboard/tables.py',
                            'def|comparison_misjudgement_step_indices': 'analysis/misjudgements.py',
                            'def|constraint_misjudgement_step_indices': 'analysis/misjudgements.py',
                            'def|create_df_no_lists': 'dashboard/tables.py',
                            'def|create_display1_df': 'dashboard/tables.py',
                            'def|create_display2_df': 'dashboard/tables.py',
                            'def|get_lon_display_columns': 'dashboard/tables.py',
                            'def|increasing_noise_step_indices': 'analysis/misjudgements.py',
                            'doc|': 'analysis/misjudgements.py'}}


# --------------------------------------------------------------- the planned Stage-11 destinations

ENTRY_MODULES = {"pre": "dashboard/Dashboard.py", "post": "dashboard/app.py"}
MLFLOW_MODULES = {"pre": "noisyvis.app.app", "post": "noisyvis.mlflow_app.app"}
MLFLOW_PAGE = {"pre": "app/pages/mlflow_browser.py", "post": "mlflow_app/pages/mlflow_browser.py"}

# Modules Stage 11 removes, and the ones it adds. Anything else appearing or disappearing from the
# running application's `sys.modules` is a finding, not a detail.
RETIRED_MODULES = (
    "noisyvis.dashboard.Dashboard", "noisyvis.dashboard.DashboardHelpers",
    "noisyvis.dataio", "noisyvis.dataio.transformers", "noisyvis.dataio.column_config",
    "noisyvis.app", "noisyvis.app.app",
)
NEW_MODULES = (
    "noisyvis.dashboard.app", "noisyvis.dashboard.instance", "noisyvis.dashboard.data",
    "noisyvis.dashboard.helpers", "noisyvis.dashboard.tables", "noisyvis.dashboard.columns",
    "noisyvis.dashboard.callbacks", "noisyvis.dashboard.callbacks.schematic",
    "noisyvis.dashboard.callbacks.selection", "noisyvis.dashboard.callbacks.performance",
    "noisyvis.dashboard.callbacks.graph_data", "noisyvis.dashboard.callbacks.visualization",
    "noisyvis.dashboard.callbacks.pareto", "noisyvis.analysis.misjudgements",
    "noisyvis.mlflow_app", "noisyvis.mlflow_app.app",
)

# Where an import's module moves. The imported *names* never change (A1).
MODULE_RENAMES = {
    "noisyvis.dataio": "noisyvis.dashboard.data",
    "noisyvis.dataio.transformers": "noisyvis.dashboard.tables",
    "noisyvis.dataio.column_config": "noisyvis.dashboard.columns",
    "noisyvis.dashboard.DashboardHelpers": "noisyvis.dashboard.helpers",
}

# `dataio.transformers` is split rather than moved (plan §6): its misjudgement analysis goes to
# `analysis.misjudgements`, the rest to `dashboard.tables`. These names override MODULE_RENAMES.
NAME_RENAMES = {
    ("noisyvis.dataio.transformers", name): "noisyvis.analysis.misjudgements"
    for name in ("_compute_n_misjudgements", "increasing_noise_step_indices",
                 "comparison_misjudgement_step_indices", "constraint_misjudgement_step_indices",
                 "_compute_n_increasing_noise", "_compute_n_comparison_misjudgements",
                 "_compute_n_constraint_misjudgements")
}

# The only imports Stage 11 removes: the seven wildcards. Every other existing explicit import is
# preserved, including the ones that are already unused (amendment A1).
REMOVED_IMPORTS = (
    ("from", "noisyvis.dashboard.DashboardHelpers", "*", None),
    ("from", "noisyvis.problems.onemax", "*", None),
    ("from", "noisyvis.problems.jump", "*", None),
    ("from", "noisyvis.problems.knapsack", "*", None),
    ("from", "noisyvis.problems.continuous", "*", None),
    ("from", "noisyvis.common.embedding", "*", None),
)

# The four names that Stage 11 rewrites onto explicit imports, with the module that supplies them.
WILDCARD_REPLACEMENTS = {
    "noisyvis.dashboard.helpers": (
        "convert_to_single_edges_format", "convert_to_split_edges_format", "filter_local_optima",
        "filter_negative_LO", "get_mean_run", "get_median_run", "select_top_runs_by_fitness"),
}

# Statements whose content may legitimately differ, with the reason. Everything else must hash equal.
ALLOWED_CHANGES = {
    ("dataio/__init__.py", "doc|"): "the package docstring becomes dashboard/data.py's",
    ("dataio/transformers.py", "doc|"): "the module docstring follows the split",
    ("app/pages/mlflow_browser.py", "cb|..tracking-uri.children...experiments-table-wrap.children.."):
        "parents[4] is replaced by results.paths.MLRUNS_DIR (plan §9)",
}

# `dataio.__all__` is the one structural deletion: a package export list, with no consumer, for a
# package that Stage 11 dissolves.
STRUCTURAL_DELETIONS = (("dataio/__init__.py", "assign|__all__"),)

# Files allowed to anchor a path by counting parents. `mlflow_browser.py` leaves this set at 11-J.
PARENTS_ANCHORS = ("results/paths.py", "app/pages/mlflow_browser.py")

# The one orchestrator: the outputs that must stay on a single callback over the shared graph (I-10c).
ORCHESTRATOR_OUTPUTS = ("trajectory-plot.figure", "run-print-info.children",
                        "stn-stats-table.children", "lon-stats-table.children",
                        "lon-feas-error-scatter.figure", "lon-selected-correlation.children",
                        "lon-feas-error-correlations.children")
ORCHESTRATOR_INPUTS = ("STN_data_processed.data", "STN_MO_data.data", "LON_data.data")


def key_str(key) -> str:
    return f'{key[0]}|{key[1] if len(key) > 1 else ""}'


def allowed_names(output: str) -> tuple:
    pre = CALLBACKS[output]["name"]
    post = inv.RENAMES.get(output, pre)
    return (pre, post) if ALLOW_PRE_LOCATIONS else (post,)


def allowed_callback_modules(output: str) -> tuple:
    post = CALLBACK_MODULES[output]
    return (ENTRY_MODULES["pre"], post) if ALLOW_PRE_LOCATIONS else (post,)


def allowed_statement_modules(pre_file: str, key) -> tuple:
    post = STATEMENT_DESTINATIONS[pre_file][key_str(key)]
    if post is None:  # a structural deletion: it exists only while the PRE file does
        return (pre_file,)
    allow_pre = (ALLOW_PRE_MLFLOW_LOCATIONS if pre_file in (inv.MLFLOW_APP_PY, inv.MLFLOW_PAGE_PY)
                 else ALLOW_PRE_LOCATIONS)
    return (pre_file, post) if allow_pre else (post,)


# --------------------------------------------------------------- the probe

_PROBE = r'''import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

SOURCE_ROOT = Path(__SOURCE_ROOT__)
WORKSPACE = Path(__WORKSPACE__)
SECTIONS = __SECTIONS__

sys.path.insert(0, str(SOURCE_ROOT / "src"))

spec = importlib.util.spec_from_file_location("_harness_fence", __FENCE__)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(str(WORKSPACE))

spec = importlib.util.spec_from_file_location("_dashboard_inventory", __INVENTORY__)
INV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INV)

import numpy as np
import pandas as pd

ROOT = Path(os.environ["NOISYVIS_ROOT"])
(ROOT / "plots").mkdir(parents=True, exist_ok=True)
PKG = SOURCE_ROOT / "src" / "noisyvis"


# ------------------------------------------------------------------ canonical forms

def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canon(obj):
    if isinstance(obj, np.bool_):
        return {"numpy.bool_": bool(obj)}
    if isinstance(obj, np.integer):
        return {"numpy." + type(obj).__name__: str(int(obj))}
    if isinstance(obj, np.floating):
        return {"numpy." + type(obj).__name__: float(obj).hex()}
    if isinstance(obj, np.ndarray):
        return {"ndarray": str(obj.dtype), "shape": list(obj.shape),
                "data": [canon(x) for x in obj.reshape(-1).tolist()]}
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return {"int": str(obj)}
    if isinstance(obj, float):
        return {"float": obj.hex()}
    if isinstance(obj, tuple):
        return {"tuple": [canon(x) for x in obj]}
    if isinstance(obj, list):
        return {"list": [canon(x) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        return {"set": sorted(repr(x) for x in obj)}
    if isinstance(obj, dict):
        return {"dict": [[canon(k), canon(v)] for k, v in obj.items()]}
    return {"other": type(obj).__qualname__, "repr": repr(obj)}


def digest(obj):
    return sha256(json.dumps(canon(obj), separators=(",", ":")))


def obj_ast_sha256(obj):
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    package = getattr(module, "__package__", "") or ""
    node = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]
    return INV.norm_ast_sha256(node, package)


def import_first(*names):
    last = None
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # noqa: PERF203
            last = exc
    raise last


def json_safe(frame):
    """Records as a dcc.Store would hold them: NaN is null."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


# ------------------------------------------------------------------ the synthetic warehouse

# The knapsack PID is a real instance: `add_lon_nodes` loads it through `load_problem_KP`, so the
# LON half of the pipeline runs for real. Ten bits, as in the Stage-10 fixtures.
KP_PID = "f1_l-d_kp_10_269"
BITS = 10
STEPS = 5


def bits(value):
    return [int(c) for c in format(value % 1024, "010b")]


def trajectory(offset, minimising):
    """Deterministic 5-visit trajectory: one decline, one negative visit, one improvement."""
    base = [10.0, 12.0, 11.0, -3.0, 15.0]
    fits = [(-value if minimising else value) + offset for value in base]
    noisy = [value + (0.5 if index % 2 == 0 else -0.5) for index, value in enumerate(fits)]
    return fits, noisy


def algo_rows():
    rows = []
    problems = [
        dict(PID=KP_PID, problem_type="knapsack", problem_goal="maximise", problem_name="kp_10",
             dimensions=BITS, opt_global=30.0, fit_func="eval_noisy_kp_v1",
             experiment_name="exp-a", experiment_description="synthetic knapsack"),
        dict(PID="om_synthetic", problem_type="onemax", problem_goal="minimise", problem_name="om_8",
             dimensions=BITS, opt_global=0.0, fit_func="OneMax_fitness",
             experiment_name=None, experiment_description=None),
    ]
    index = 0
    for problem in problems:
        minimising = problem["problem_goal"] == "minimise"
        for algo in ("MuPlusLambda", "OnePlusOne"):
            for noise in (0.0, 1.0):
                for seed in (1, 2):
                    index += 1
                    fits, noisy = trajectory(float(index), minimising)
                    sols = [bits(index * 7 + step * 11) for step in range(STEPS)]
                    row = dict(problem)
                    row.update(
                        algo_type="SO", algo_name=algo, noise=noise, seed=seed,
                        seed_signature=f"sig-{index}", n_gens=10 + index, n_evals=100 + index,
                        stop_trigger="eval_limit", n_unique_sols=STEPS,
                        final_fit=fits[-1], max_fit=max(fits), min_fit=min(fits),
                        penalty=float(index % 2), peak_ram_mb=100.0 + index,
                        rep_sols=sols, rep_fits=fits, rep_noisy_fits=noisy,
                        rep_noisy_sols=sols, rep_fitness_boxplot_stats=[],
                        rep_estimated_fits_whenadopted=[], rep_estimated_fits_whendiscarded=[],
                        count_estimated_fits_whenadopted=[], count_estimated_fits_whendiscarded=[],
                        sol_iterations=[1, 2, 1, 1, 3], sol_iterations_evals=[2, 4, 2, 2, 6],
                        # pairs of (previous, current) solutions, as the logger records them
                        sol_transitions=[[sols[i], sols[i + 1]] for i in range(STEPS - 1)],
                        alternative_rep_sols=[], alternative_rep_fits=[],
                    )
                    rows.append(row)

    # One multi-objective row: rep_fits is NaN, as in the real warehouse, with Pareto columns.
    mo = dict(problems[0])
    mo.update(
        algo_type="MO", algo_name="NSGA2", noise=1.0, seed=1, seed_signature="sig-mo",
        n_gens=3, n_evals=90, stop_trigger="gen_limit", n_unique_sols=0,
        final_fit=np.nan, max_fit=np.nan, min_fit=np.nan, penalty=0.0, peak_ram_mb=120.0,
        rep_sols=None, rep_fits=np.nan, rep_noisy_fits=np.nan, rep_noisy_sols=None,
        rep_fitness_boxplot_stats=[], rep_estimated_fits_whenadopted=[],
        rep_estimated_fits_whendiscarded=[], count_estimated_fits_whenadopted=[],
        count_estimated_fits_whendiscarded=[], sol_iterations=None,
        sol_iterations_evals=None, sol_transitions=None,
        alternative_rep_sols=[], alternative_rep_fits=[],
        pareto_solutions=[[bits(3), bits(5)], [bits(9), bits(12)]],
        pareto_fitnesses=[[[1.0, 2.0], [2.0, 1.0]], [[1.5, 2.5], [2.5, 1.5]]],
        pareto_true_fitnesses=[[[1.1, 2.1], [2.1, 1.1]], [[1.6, 2.6], [2.6, 1.6]]],
        true_pareto_solutions=[[bits(4), bits(6)], [bits(10), bits(13)]],
        true_pareto_fitnesses=[[[1.2, 2.2], [2.2, 1.2]], [[1.7, 2.7], [2.7, 1.7]]],
        noisy_pf_noisy_hypervolumes=[3.0, 4.0], noisy_pf_true_hypervolumes=[3.1, 4.1],
        true_pf_hypervolumes=[3.2, 4.2], n_gens_pareto_best=[1, 2],
    )
    rows.append(mo)
    return rows


def lon_rows():
    optima = [bits(v) for v in (3, 15, 85, 170)]
    edges = {}
    for i in range(len(optima)):
        source = tuple(optima[i])
        target = tuple(optima[(i + 1) % len(optima)])
        edges[(source, target)] = i + 1
    return [dict(
        PID=KP_PID, problem_name="kp_10", problem_type="knapsack", problem_goal="maximise",
        dimensions=BITS, opt_global=30.0, fit_func="eval_noisy_kp_v1", algo_name="ILS",
        noise=float(noise), n_runs=5, iterations=20,
        local_optima=optima, fitness_values=[10.0, 12.0, 14.0, 16.0],
        edges=edges, optima_feasibility=[1, 0, 1, 1],
        neighbour_feasibility=[0.25, 0.5, 0.75, 1.0],
        visit_counts=[4, 3, 2, 1], visit_proportions=[0.4, 0.3, 0.2, 0.1],
    ) for noise in (0, 1)]


def write_fixture():
    warehouse = ROOT / "data" / "warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)
    algo = pd.DataFrame(algo_rows())
    lon = pd.DataFrame(lon_rows())
    algo.to_pickle(warehouse / "algo_results.pkl")
    lon.to_pickle(warehouse / "lon_results.pkl")
    return algo, lon


# ------------------------------------------------------------------ the app under test

LOAD_COUNT = {"algo": 0, "lon": 0}


def load_entry_module():
    store = importlib.import_module("noisyvis.results.store")
    for kind in ("algo", "lon"):
        real = getattr(store, f"load_{kind}_results")

        def counting(*args, _real=real, _kind=kind, **kwargs):
            LOAD_COUNT[_kind] += 1
            return _real(*args, **kwargs)

        setattr(store, f"load_{kind}_results", counting)
    return import_first("noisyvis.dashboard.app", "noisyvis.dashboard.Dashboard")


def callback_functions(app):
    """output id -> the undecorated function Dash registered."""
    out = {}
    for spec in app._callback_list:
        wrapper = app.callback_map[spec["output"]]["callback"]
        out[spec["output"]] = getattr(wrapper, "__wrapped__", wrapper)
    return out


def source_module(func):
    path = Path(inspect.getsourcefile(func)).resolve()
    return str(path.relative_to(PKG))


def callback_report(app):
    report = {}
    functions = callback_functions(app)
    for order, spec in enumerate(app._callback_list):
        output = spec["output"]
        func = functions[output]
        node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
        module = sys.modules[func.__module__]
        package = getattr(module, "__package__", "") or ""
        # The AST is hashed under a fixed name, so a B9 rename does not change it; the name itself
        # is pinned separately, against the frozen PRE name or its planned replacement.
        hashed = copy.deepcopy(node)
        hashed.name = "CALLBACK"
        report[output] = {
            "order": order,
            "name": func.__name__,
            "module": source_module(func),
            "params": list(inspect.signature(func).parameters),
            "ast": INV.norm_ast_sha256(hashed, package),
            "literals": INV.literal_digest(hashed),
            "inputs": [f'{i["id"]}.{i["property"]}' for i in spec["inputs"]],
            "state": [f'{s["id"]}.{s["property"]}' for s in spec["state"]],
            "prevent_initial_call": spec["prevent_initial_call"],
        }
    return report


def global_bindings(*modules):
    """Every module global in the app modules that holds one of the loaded DataFrames."""
    bindings = {}
    for module in modules:
        for name, value in vars(module).items():
            for attr, obj in DATA_IDENTITY.items():
                if value is obj:
                    bindings[f"{module.__name__}:{name}"] = attr
    return dict(sorted(bindings.items()))


def free_name_report(app):
    """Every module global each callback reads, resolved to the object it names."""
    functions = callback_functions(app)
    report = {}
    for output, func in functions.items():
        node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
        resolved = {}
        for name in sorted(INV.free_names(node)):
            if name in ("Input", "Output", "State", "app"):
                continue
            value = func.__globals__.get(name, KeyError)
            resolved[name] = describe(value)
        report[output] = resolved
    return report


DATA_ATTRS = ("df", "df_lon", "df_no_lists", "display1_df", "display2_df", "lon_display_columns")
DATA_IDENTITY = {}


def describe(value):
    if value is KeyError:
        return {"kind": "UNRESOLVED"}
    for attr, obj in DATA_IDENTITY.items():
        if value is obj:
            return {"kind": "data", "attr": attr}
    if inspect.ismodule(value):
        return {"kind": "module", "name": value.__name__}
    if isinstance(value, type):
        return {"kind": "class", "name": value.__name__, "ast": obj_ast_sha256(value)}
    if callable(value):
        try:
            return {"kind": "callable", "name": value.__name__, "ast": obj_ast_sha256(value)}
        except (OSError, TypeError):
            return {"kind": "callable", "name": getattr(value, "__name__", repr(value))}
    if isinstance(value, pd.DataFrame):
        return {"kind": "frame", "columns": list(value.columns), "shape": list(value.shape)}
    described = canon(value)
    if isinstance(described, dict) and "other" in described:
        # repr() of an opaque object carries its address; the type is the contract.
        return {"kind": "object", "type": f"{type(value).__module__}.{type(value).__qualname__}"}
    return {"kind": "value", "canon": described}


# ------------------------------------------------------------------ the HTTP driver

def split_callback_id(callback_id):
    if callback_id.startswith(".."):
        return [split_callback_id(part) for part in callback_id[2:-2].split("...")]
    id_, prop = callback_id.rsplit(".", 1)
    return {"id": id_, "property": prop}


def build_values(data):
    """A value for every (id, property) the 41 callbacks read, derived from the fixture."""
    display1 = json_safe(data.display1_df)
    display2 = json_safe(data.display2_df)
    plot_2d = json_safe(data.df_no_lists)
    lon_table = json_safe(data.df_lon[data.lon_display_columns])
    stn_rows = json_safe(
        data.df[(data.df["algo_name"] == "MuPlusLambda") & (data.df["noise"] == 1.0)])
    values = {
        "experiment-selector.value": "exp-a",
        "table1.data": display1,
        "table1.selected_rows": [0],
        "table1-selected-store.data": [0],
        "table2.data": display2,
        "table2.selected_rows": [0],
        "table2.derived_virtual_data": display2,
        "data-problem-specific.data": display2,
        "plot_2d_data.data": plot_2d,
        "PID.data": KP_PID,
        "optimum.data": 30.0,
        "opt_goal.data": "maximise",
        "fit_func_store.data": "eval_noisy_kp_v1",
        "lon-table-selected-pid-store.data": KP_PID,
        "LON_table.data": lon_table,
        "LON_table.selected_rows": [0],
        "LON_data.data": None,
        "STN_data.data": stn_rows,
        "STN_data_processed.data": [],
        "STN_series_labels.data": [],
        "STN_MO_data.data": [],
        "STN_MO_series_labels.data": [],
        "MO_data_PPP.data": [],
        "noisy_fitnesses_data.data": [],
        "mo_plot_type.value": "npnhv",
        "penalty-filter-dropdown.value": None,
        "so-fitness-mode.value": "best",
        "plot-theme.value": "Viridis",
        "noise-cap-input.value": None,
        "hide-series-dropdown.value": [],
        "advanced-misjudgement-algo-dropdown.value": "MuPlusLambda",
        "line-evals-show-std.value": ["show"],
        "round-stats-checkbox.value": ["round"],
        "scientific-notation-checkbox.value": [],
        "2DPlotTabSelection.value": "p1",
        "schematic-misjudgements.value": ["misjudgements"],
        "schematic-simple-annotations.value": [],
        "schematic-boxplots.value": ["boxplots"],
        "show-text-info.value": ["show"],
        "annotation-options.value": ["annotate-start-nodes", "annotate-end-nodes"],
        "annotation-text-scale.value": 1.0,
        "axes-text-scale.value": 1.0,
        "info-panel-x.value": 90,
        "info-panel-y.value": 75,
        "custom_x_min.value": None, "custom_x_max.value": None,
        "custom_y_min.value": None, "custom_y_max.value": None,
        "custom_z_min.value": None, "custom_z_max.value": None,
        "log-z-axis.value": [],
        "axis-values.data": {"custom_x_min": None, "custom_x_max": None, "custom_y_min": None,
                             "custom_y_max": None, "custom_z_min": None, "custom_z_max": None,
                             "log_z": False},
        "options.value": ["plot_3D"],
        "run-options.value": ["use-viridis"],
        "run-index.value": 0,
        "run-selector.value": 1,
        "STN_lower_fit_limit.value": None,
        "LON-fit-percent.value": 100,
        "LON-options.value": [],
        "LON-node-colour-mode.value": "fitness",
        "LON-surface-colour.value": "fitness",
        "LON-edge-colour-feas.value": [],
        "lmds-multiplier.value": 1.0,
        "NLON_fit_func.value": "",
        "NLON_intensity.value": 1,
        "NLON_samples.value": 10,
        "NLON_penalty.value": 10,
        "layout.value": "mds",
        "plotType.value": "RegLon",
        "hover-info.value": "fitness",
        "azimuth_deg.value": 35,
        "elevation_deg.value": 60,
        "opacity_noise_bar.value": 1,
        "LON_node_opacity.value": 1,
        "LON_edge_opacity.value": 1,
        "STN_node_opacity.value": 1,
        "STN_edge_opacity.value": 1,
        "STN-node-min.value": 5,
        "STN-node-max.value": 20,
        "LON-node-min.value": 10,
        "LON-node-max.value": 10.1,
        "LON-edge-size-slider.value": 5,
        "STN-edge-size-slider.value": 5,
        "stn-plot-type.value": "posterior",
        "stn-node-size-metric.value": "generations",
        "lon-scatter-x-axis.value": "neigh_feas",
        "lon-scatter-y-axis.value": "error",
        "lon-scatter-plot-style.value": "scatter",
        "lon-scatter-multi-noise.value": [],
        "paretoFrontPlotType.value": "Basic",
        "IndVsDist_IndType.value": "hypervolume",
        "IndVsDist_DistType.value": "euclidean",
        "paretoPlotNumRuns.value": 1,
        "paretoPlotWindowSize.value": 2,
    }
    return values


def call(client, spec, values):
    """One `/_dash-update-component` POST, exactly as the browser makes it."""
    output = spec["output"]
    body = {
        "output": output,
        "outputs": split_callback_id(output),
        "inputs": [{"id": i["id"], "property": i["property"],
                    "value": values.get(f'{i["id"]}.{i["property"]}')}
                   for i in spec["inputs"]],
        "state": [{"id": s["id"], "property": s["property"],
                   "value": values.get(f'{s["id"]}.{s["property"]}')}
                  for s in spec["state"]],
        "changedPropIds": [f'{spec["inputs"][0]["id"]}.{spec["inputs"][0]["property"]}'],
    }
    response = client.post("/_dash-update-component", json=body)
    text = response.get_data(as_text=True)
    entry = {"status": response.status_code}
    payload = None
    if response.status_code == 200:
        payload = json.loads(text)
        entry["digest"] = digest(payload)
        entry["props"] = sorted(f"{cid}.{prop}"
                                for cid, props in payload.get("response", {}).items()
                                for prop in props)
        entry["length"] = len(text)
        entry["traces"] = {f"{cid}.{prop}": len(value["data"])
                           for cid, props in payload.get("response", {}).items()
                           for prop, value in props.items()
                           if isinstance(value, dict) and isinstance(value.get("data"), list)}
    else:
        entry["body"] = text[:200]
    return entry, payload


def http_report(app, values):
    client = app.server.test_client()
    report = {}
    for spec in app._callback_list:
        report[spec["output"]], _ = call(client, spec, values)
    return report


# The populated path: selection -> STN/LON stores -> the one shared-graph orchestrator (I-10c).
CHAIN = ["data-problem-specific.data",
         "..optimum.data...PID.data...opt_goal.data...fit_func_store.data..",
         "plot_2d_data.data",
         "STN_data.data",
         "..STN_data_processed.data...STN_series_labels.data...noisy_fitnesses_data.data..."
         "STN_MO_data.data...STN_MO_series_labels.data...MO_data_PPP.data..",
         "LON_data.data",
         "..trajectory-plot.figure...run-print-info.children...stn-stats-table.children..."
         "lon-stats-table.children...lon-feas-error-scatter.figure..."
         "lon-selected-correlation.children...lon-feas-error-correlations.children.."]

CHAIN_FEEDS = {
    "data-problem-specific.data": [("data-problem-specific", "data")],
    "plot_2d_data.data": [("plot_2d_data", "data")],
    "STN_data.data": [("STN_data", "data")],
    "LON_data.data": [("LON_data", "data")],
}


def chain_report(app, values):
    """Drive the real pipeline in order, feeding each response into the next request."""
    app.server.config["PROPAGATE_EXCEPTIONS"] = os.environ.get("PROBE_DEBUG") == "1"
    client = app.server.test_client()
    specs = {spec["output"]: spec for spec in app._callback_list}
    values = dict(values)
    report = {}
    for output in CHAIN:
        if output not in specs:
            # The pipeline no longer has this callback: report it rather than raising, so the
            # orchestrator test explains what changed.
            report[output] = {"status": "MISSING"}
            continue
        entry, payload = call(client, specs[output], values)
        report[output] = entry
        if payload is None:
            continue
        for component, props in payload.get("response", {}).items():
            for prop, value in props.items():
                values[f"{component}.{prop}"] = value
    return report


# ------------------------------------------------------------------ import surface

def import_surface():
    modules = sorted(name for name in sys.modules if name.startswith("noisyvis"))
    top_level = sorted({name.split(".")[0] for name in sys.modules
                        if not name.startswith("_") and "." not in name})
    statements = {}
    for path in sorted(PKG.rglob("*.py")):
        key = str(path.relative_to(PKG))
        if key.startswith(("dashboard/", "dataio/", "app/", "mlflow_app/", "analysis/")):
            statements[key] = INV.import_statements(PKG, path)
    return {
        "noisyvis_modules": modules,
        "top_level_packages": top_level,
        "statements": statements,
    }


# ------------------------------------------------------------------ table schemas

def frame_report(frame):
    return {
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "shape": list(frame.shape),
        "digest": digest(json_safe(frame)),
    }


def tables_report(algo, lon):
    tables = import_first("noisyvis.dashboard.tables", "noisyvis.dataio.transformers")
    columns = import_first("noisyvis.dashboard.columns", "noisyvis.dataio.column_config")
    report = {
        "df_no_lists": frame_report(tables.create_df_no_lists(algo)),
        "display1_df": frame_report(tables.create_display1_df(algo)),
        "display2_df": frame_report(tables.create_display2_df(algo)),
        "lon_display_columns": tables.get_lon_display_columns(lon),
        "constants": {name: getattr(columns, name) for name in (
            "LON_HIDDEN_COLUMNS", "DISPLAY2_HIDDEN_COLUMNS", "DISPLAY1_COLUMNS",
            "LIST_COLUMNS", "DISPLAY2_DROP_COLUMNS", "DISPLAY2_DEDUP_KEYS")},
    }
    misjudgements = import_first("noisyvis.analysis.misjudgements", "noisyvis.dataio.transformers")
    fits = [10.0, 12.0, 11.0, -3.0, 15.0]
    noisy = [10.5, 11.5, 12.0, -2.5, 14.5]
    report["step_indices"] = {
        "increasing_noise": misjudgements.increasing_noise_step_indices(fits, noisy),
        "comparison_max": misjudgements.comparison_misjudgement_step_indices(fits, noisy, False),
        "comparison_min": misjudgements.comparison_misjudgement_step_indices(fits, noisy, True),
        "constraint": misjudgements.constraint_misjudgement_step_indices(fits),
    }
    return report


def script_report():
    """Do the declared console scripts resolve to a callable? (They cannot until Stage 11 lands.)"""
    from importlib.metadata import entry_points

    report = {}
    for entry in entry_points(group="console_scripts"):
        if not entry.name.startswith("noisyvis-"):
            continue
        try:
            target = entry.load()
            report[entry.name] = "callable" if callable(target) else type(target).__name__
        except Exception as exc:  # noqa: BLE001
            report[entry.name] = type(exc).__name__
    return dict(sorted(report.items()))


# ------------------------------------------------------------------ MLflow browser app

def mlflow_report():
    import dash

    module = import_first("noisyvis.mlflow_app.app", "noisyvis.app.app")
    app = module.app
    client = app.server.test_client()
    index = client.get("/")
    deps = json.loads(client.get("/_dash-dependencies").get_data(as_text=True))
    registry = {
        key: {"path": page["path"], "name": page["name"], "module": page["module"],
              "relative_path": page["relative_path"]}
        for key, page in dash.page_registry.items()
    }
    page_module = sys.modules["pages.mlflow_browser"]
    return {
        "module": module.__name__,
        "index_status": index.status_code,
        "registry": registry,
        "dependencies": [entry["output"] for entry in deps],
        "dependencies_digest": digest(deps),
        "layout_digest": digest(app.layout.to_plotly_json()),
        "page_file": str(Path(page_module.__file__).resolve().relative_to(PKG)),
        "page_globals": sorted(name for name in vars(page_module) if not name.startswith("_")),
        "config": {"use_pages": app.use_pages,
                   "assets_folder": str(Path(app.config.assets_folder).relative_to(SOURCE_ROOT)),
                   "pages_folder": str(Path(app.config.pages_folder).relative_to(SOURCE_ROOT))},
    }


# ------------------------------------------------------------------ main

def main():
    report = {"meta": {"python": sys.version.split()[0],
                       "source_root": str(SOURCE_ROOT),
                       "hash_seed": os.environ.get("PYTHONHASHSEED")}}

    if "app" in SECTIONS:
        algo, lon = write_fixture()
        entry = load_entry_module()
        app = entry.app
        report["meta"]["entry_module"] = entry.__name__

        data_module = import_first("noisyvis.dashboard.data", "noisyvis.dataio")
        holder = entry if hasattr(entry, "data") else data_module
        for attr in DATA_ATTRS:
            DATA_IDENTITY[attr] = getattr(holder.data, attr)

        report["load_count"] = dict(LOAD_COUNT)
        report["callbacks"] = callback_report(app)
        report["callback_list_digest"] = digest(app._callback_list)
        # The same digest the live /_dash-dependencies endpoint produces.
        report["dependencies_digest"] = sha256(
            json.dumps(app._callback_list, sort_keys=True))
        report["callback_order"] = [spec["output"] for spec in app._callback_list]
        report["free_names"] = free_name_report(app)
        report["config"] = {
            "suppress_callback_exceptions": app.config.suppress_callback_exceptions,
            "prevent_initial_callbacks": app.config.prevent_initial_callbacks,
            "assets_folder": str(Path(app.config.assets_folder).relative_to(SOURCE_ROOT)),
            "assets_url_path": app.config.assets_url_path,
            "serve_locally": app.config.serve_locally,
            "include_assets_files": app.config.include_assets_files,
            "title": app.title,
            "update_title": app.config.update_title,
            "url_base_pathname": app.config.url_base_pathname,
            "use_pages": app.use_pages,
            "assets_folder_exists": Path(app.config.assets_folder).is_dir(),
        }
        report["layout_digest"] = digest(app.layout.to_plotly_json())
        report["layout_ids"] = sorted(
            component.id for component in app.layout._traverse()
            if getattr(component, "id", None) is not None)
        report["globals"] = {
            "data_module": data_module.__name__,
            "holder": holder.__name__,
            "bindings": global_bindings(entry, data_module),
        }
        values = build_values(holder.data)
        report["http"] = http_report(app, values)
        report["chain"] = chain_report(app, values)
        report["tables"] = tables_report(algo, lon)
        report["import_surface"] = import_surface()
        report["scripts"] = script_report()
        report["entry_main"] = {
            "has_main": callable(getattr(entry, "main", None)),
            "module_file": str(Path(entry.__file__).resolve().relative_to(PKG)),
        }

    if "mlflow" in SECTIONS:
        report["mlflow"] = mlflow_report()

    report["meta"]["noisyvis_file"] = importlib.import_module("noisyvis").__file__
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))


main()
'''


def build_probe(source_root, sections) -> str:
    replacements = {
        "__SOURCE_ROOT__": repr(str(source_root)),
        "__WORKSPACE__": repr(str(WORKSPACE)),
        "__FENCE__": repr(str(HARNESS_DIR / "fence.py")),
        "__INVENTORY__": repr(str(INVENTORY_PATH)),
        "__SECTIONS__": repr(list(sections)),
    }
    probe = _PROBE
    for marker, value in replacements.items():
        probe = probe.replace(marker, value)
    return probe


def _run_probe(sections) -> dict:
    """One probe subprocess, in a fresh temp root, with the write fence installed.

    The production warehouse is never read: the probe writes its own synthetic one into the root that
    `NOISYVIS_ROOT` points at, and the real loader reads that.
    """
    root = make_temp_root()
    env = child_env(root)
    env["PYTHONHASHSEED"] = "0"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", build_probe(WORKSPACE, sections)],
            cwd=str(root), env=env, capture_output=True, text=True, timeout=1800,
        )
        assert completed.returncode == 0, (
            f"dashboard probe {sections} failed:\n{completed.stderr[-4000:]}")
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert report["meta"]["noisyvis_file"].startswith(str(WORKSPACE / "src" / "noisyvis")), \
        report["meta"]
    assert report["meta"]["hash_seed"] == "0", report["meta"]
    return report


@pytest.fixture(scope="module")
def app_probe() -> dict:
    """The dashboard application: contract, config, layout, HTTP, tables, import surface."""
    return _run_probe(["app"])


@pytest.fixture(scope="module")
def mlflow_probe() -> dict:
    """The MLflow browser application."""
    return _run_probe(["mlflow"])


def _import_fails(module: str) -> str:
    """Import `module` in a fresh interpreter; return the exception class name, or "IMPORTED".

    The tree under test comes first on the path, so this answers for that tree rather than for
    whatever the interpreter would otherwise resolve.
    """
    lines = ["import importlib, sys", f"sys.path.insert(0, {str(PKG.parent)!r})",
             "try:", f"    importlib.import_module({module!r})",
             "    print('IMPORTED')", "except Exception as exc:", "    print(type(exc).__name__)"]
    completed = subprocess.run([sys.executable, "-c", chr(10).join(lines)],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip()


def _mismatches(observed: dict, expected: dict) -> list:
    keys = sorted(set(observed) | set(expected))
    return [f"{key}:\n    expected {expected.get(key)!r}\n    observed {observed.get(key)!r}"
            for key in keys if observed.get(key) != expected.get(key)]


# ------------------------------------------------------------------------------ the tests


def test_callback_contract_pinned(app_probe):
    """41 callbacks, in one order, with the same Outputs, Inputs, States and options."""
    assert app_probe["callback_order"] == CALLBACK_ORDER, (
        "the callback registration order changed:\n"
        f"  expected {CALLBACK_ORDER}\n  observed {app_probe['callback_order']}"
    )
    assert app_probe["callback_list_digest"] == CALLBACK_LIST_DIGEST
    assert app_probe["dependencies_digest"] == DEPENDENCIES_DIGEST, (
        "the /_dash-dependencies payload changed; the browser contract is not what it was"
    )

    observed = app_probe["callbacks"]
    assert set(observed) == set(CALLBACKS), (
        f"  missing: {sorted(set(CALLBACKS) - set(observed))}\n"
        f"  added:   {sorted(set(observed) - set(CALLBACKS))}"
    )
    for output in CALLBACK_ORDER:
        got, expected = observed[output], CALLBACKS[output]
        for field in ("order", "inputs", "state", "prevent_initial_call", "params",
                      "ast", "literals"):
            assert got[field] == expected[field], (
                f"{output}: {field} changed\n    expected {expected[field]!r}\n"
                f"    observed {got[field]!r}"
            )
        assert got["name"] in allowed_names(output), (
            f"{output}: callback named {got['name']}, expected one of {allowed_names(output)}"
        )
        assert got["module"] in allowed_callback_modules(output), (
            f"{output}: defined in {got['module']}, expected one of "
            f"{allowed_callback_modules(output)}"
        )


def test_duplicate_callback_names_never_grow(app_probe):
    """B9: the shadowed names are the four frozen ones, and the renames remove them."""
    seen = {}
    for output, entry in app_probe["callbacks"].items():
        seen.setdefault((entry["module"], entry["name"]), []).append(output)
    duplicates = {name: sorted(outputs) for (module, name), outputs in seen.items()
                  if len(outputs) > 1}

    for name, outputs in duplicates.items():
        assert name in B9_DUPLICATES, f"a new shadowed callback name appeared: {name} -> {outputs}"
        assert set(outputs) <= set(B9_DUPLICATES[name]), f"{name}: {outputs}"

    for output, new_name in inv.RENAMES.items():
        if app_probe["callbacks"][output]["name"] == new_name:
            assert not any(output in outputs for outputs in duplicates.values()), (
                f"{output} was renamed to {new_name} but is still shadowed"
            )
    if not ALLOW_PRE_LOCATIONS:
        assert duplicates == {}, f"B9 is not resolved: {duplicates}"


def test_shared_graph_orchestrator_is_single(app_probe):
    """I-10c: one conceptual orchestrator over one shared graph, never an STN and a LON pipeline."""
    owners = {}
    for output, entry in app_probe["callbacks"].items():
        for target in (output[2:-2].split("...") if output.startswith("..") else [output]):
            owners.setdefault(target, []).append(output)

    orchestrators = {owners[target][0] for target in ORCHESTRATOR_OUTPUTS}
    assert len(orchestrators) == 1, (
        "the shared-graph outputs are served by more than one callback; Stage 11 must not split "
        f"update_plot into independent STN/LON pipelines: {sorted(orchestrators)}"
    )
    orchestrator = orchestrators.pop()
    for target in ORCHESTRATOR_OUTPUTS:
        assert owners[target] == [orchestrator], f"{target} is also produced by {owners[target]}"

    inputs = app_probe["callbacks"][orchestrator]["inputs"]
    for required in ORCHESTRATOR_INPUTS:
        assert required in inputs, (
            f"the orchestrator no longer reads {required}; the STN and LON halves must reach the "
            "same callback"
        )

    changed = _mismatches(app_probe["chain"], CHAIN)
    assert not changed, (
        "the populated pipeline (selection -> stores -> shared graph) changed:\n  "
        + "\n  ".join(changed)
    )
    figure = app_probe["chain"][orchestrator]["traces"]["trajectory-plot.figure"]
    assert figure == CHAIN[orchestrator]["traces"]["trajectory-plot.figure"] > 0


def test_callback_free_names_resolve(app_probe):
    """Every global each callback reads still names the same object (the wildcard contract)."""
    observed = app_probe["free_names"]
    assert sorted(observed) == sorted(FREE_NAMES)
    for output in sorted(FREE_NAMES):
        unresolved = [name for name, value in observed[output].items()
                      if value.get("kind") == "UNRESOLVED"]
        assert not unresolved, f"{output}: unresolved names {unresolved}"
        changed = _mismatches(observed[output], FREE_NAMES[output])
        assert not changed, f"{output}: resolved globals changed:\n  " + "\n  ".join(changed)


def test_callback_http_contract(app_probe):
    """Every callback, driven over the real Dash protocol, returns what it returned before."""
    observed = app_probe["http"]
    assert sorted(observed) == sorted(HTTP)
    changed = _mismatches(observed, HTTP)
    assert not changed, "callback responses changed:\n  " + "\n  ".join(changed)
    assert {entry["status"] for entry in observed.values()} == {200}


def test_app_config_layout_and_single_load(app_probe):
    """The Dash configuration, the layout, and one module-scope data load per import (R9)."""
    changed = _mismatches(app_probe["config"], CONFIG)
    assert not changed, "the Dash app configuration changed:\n  " + "\n  ".join(changed)

    assert app_probe["layout_digest"] == LAYOUT_DIGEST, "the rendered layout changed"
    assert app_probe["layout_ids"] == LAYOUT_IDS, (
        f"  missing: {sorted(set(LAYOUT_IDS) - set(app_probe['layout_ids']))}\n"
        f"  added:   {sorted(set(app_probe['layout_ids']) - set(LAYOUT_IDS))}"
    )
    assert app_probe["load_count"] == LOAD_COUNT, (
        "DashboardData.load() must run exactly once per application import (R9); "
        f"observed {app_probe['load_count']}"
    )

    # The frames stay module globals under the same names, and are the same objects (A1/R9).
    bindings = {name.split(":")[1]: attr for name, attr in app_probe["globals"]["bindings"].items()}
    assert bindings == {name.split(":")[1]: attr for name, attr in GLOBALS["bindings"].items()}


def test_table_schemas_pinned(app_probe):
    """DataFrame columns, order, dtypes and content, and the column constants."""
    observed = app_probe["tables"]
    changed = _mismatches(observed, TABLES)
    assert not changed, "the dashboard table schemas changed:\n  " + "\n  ".join(changed)


def test_import_surface_pinned(app_probe):
    """A1: the wildcards go, every other existing import stays, and nothing changes at import time."""
    observed = app_probe["import_surface"]

    assert observed["top_level_packages"] == IMPORT_SURFACE["top_level_packages"], (
        "the set of loaded third-party packages changed; Stage 11 must have no import-time delta "
        "(amendment A1):\n"
        f"  missing: {sorted(set(IMPORT_SURFACE['top_level_packages']) - set(observed['top_level_packages']))}\n"
        f"  added:   {sorted(set(observed['top_level_packages']) - set(IMPORT_SURFACE['top_level_packages']))}"
    )

    before, now = set(IMPORT_SURFACE["noisyvis_modules"]), set(observed["noisyvis_modules"])
    assert (before - now) <= set(RETIRED_MODULES), (
        f"modules stopped loading that Stage 11 does not retire: {sorted(before - now)}"
    )
    assert (now - before) <= set(NEW_MODULES), (
        f"unexpected modules now load: {sorted(now - before)}"
    )

    def rename(statement):
        kind, module, *rest = statement
        if kind == "from" and (module, rest[0]) in NAME_RENAMES:
            return tuple([kind, NAME_RENAMES[(module, rest[0])], *rest])
        return tuple([kind, MODULE_RENAMES.get(module, module), *rest])

    expected = ({rename(statement) for statements in IMPORT_SURFACE["statements"].values()
                 for statement in statements}
                - {rename(list(item)) for item in REMOVED_IMPORTS}
                - {tuple(item) for item in REMOVED_IMPORTS})
    present = {rename(statement) for statements in observed["statements"].values()
               for statement in statements}
    missing = sorted(expected - present)
    assert not missing, (
        "explicit imports disappeared; A1 keeps every one of them, even the unused ones:\n  "
        + "\n  ".join(map(str, missing))
    )

    # The wildcards may only be replaced by explicit imports of the names they actually supplied.
    wildcards = [statement for statements in observed["statements"].values()
                 for statement in statements if statement[-2:] == ["*", None]]
    if not ALLOW_PRE_LOCATIONS:
        assert wildcards == [], f"a wildcard import survives: {wildcards}"
        for module, names in WILDCARD_REPLACEMENTS.items():
            imported = {statement[2] for statements in observed["statements"].values()
                        for statement in statements
                        if statement[0] == "from" and statement[1] == module}
            assert set(names) <= imported, (
                f"{module}: consumed wildcard names are not imported explicitly: "
                f"{sorted(set(names) - imported)}"
            )


def test_statement_inventory(app_probe):
    """Every statement of the seven redistributed files: exactly once, in an approved module."""
    constants = inv.store_constants(PKG)
    for pre_file, frozen in STATEMENTS.items():
        located = inv.locate(PKG, pre_file, frozen, constants)
        pre_exists = (PKG / pre_file).is_file()

        for entry, record in zip(frozen, located):
            key = tuple(entry["key"])
            label = f"{pre_file}:{key_str(key)}"
            if (pre_file, key_str(key)) in STRUCTURAL_DELETIONS and not pre_exists:
                assert record["count"] == 0, f"{label}: expected the enumerated deletion"
                continue
            assert record["count"] == 1, (
                f"{label}: found {record['count']} times in {record['found_in']}"
            )
            assert record["found_in"] in allowed_statement_modules(pre_file, key), (
                f"{label}: in {record['found_in']}, expected one of "
                f"{allowed_statement_modules(pre_file, key)}"
            )
            if (pre_file, key_str(key)) in ALLOWED_CHANGES:
                continue
            assert record["ast"] == entry["ast"], f"{label}: the statement's AST changed"
            assert record["literals"] == entry["literals"], f"{label}: its string literals changed"

        if not pre_exists:
            module = "noisyvis." + pre_file.replace("/__init__.py", "").replace(".py", "").replace("/", ".")
            assert _import_fails(module) == "ModuleNotFoundError", (
                f"{pre_file} is gone but {module} is still importable"
            )


def test_entrypoints_and_console_scripts(app_probe):
    """`main()`, the declared console scripts, and the Compose commands they must match."""
    entry = app_probe["entry_main"]
    assert entry["module_file"] in (
        (ENTRY_MODULES["pre"], ENTRY_MODULES["post"]) if ALLOW_PRE_LOCATIONS
        else (ENTRY_MODULES["post"],))

    # Each console script is checked against its own app's stage. SCRIPTS is the PRE state, in which
    # neither target module existed yet.
    expected_scripts = dict(SCRIPTS)
    if entry["module_file"] == ENTRY_MODULES["post"]:
        assert entry["has_main"], "noisyvis.dashboard.app must expose main()"
        expected_scripts["noisyvis-dashboard"] = "callable"
    if not ALLOW_PRE_MLFLOW_LOCATIONS:
        expected_scripts["noisyvis-mlflow"] = "callable"
    assert app_probe["scripts"] == expected_scripts, (
        "a console-script target does not match its app's stage: "
        f"expected {expected_scripts}, observed {app_probe['scripts']}"
    )

    # Every `python -m ...` a committed Compose file runs must exist in this tree.
    for compose in sorted((WORKSPACE / "docker").glob("compose.*.yaml")):
        text = compose.read_text()
        for match in re.finditer(r'"python"\s*,\s*"-m"\s*,\s*"([\w.]+)"', text):
            module = match.group(1)
            relative = Path(*module.split(".")[1:])
            candidates = [PKG / relative.with_suffix(".py"), PKG / relative / "__init__.py"]
            assert any(path.is_file() for path in candidates), (
                f"{compose.name} runs `python -m {module}`, which does not exist in this tree"
            )


def test_entrypoint_modules_are_never_imported():
    """Under `-m` an entrypoint module is `__main__`; importing it would load a second app."""
    offenders = []
    for path in sorted((WORKSPACE / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        package = inv.package_of(PKG, path) if str(path).startswith(str(PKG)) else ""
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                resolved = inv.absolutise(node, package) if package else node
                modules = [resolved.module or ""]
                modules += [f"{resolved.module}.{alias.name}" for alias in node.names]
            for module in modules:
                if module in ("noisyvis.dashboard.app", "noisyvis.mlflow_app.app"):
                    offenders.append(f"{path.relative_to(WORKSPACE)}:{node.lineno} -> {module}")
    assert not offenders, "an entrypoint module is imported:\n  " + "\n  ".join(offenders)


def test_single_filesystem_anchor():
    """`results/paths.py` is the path anchor; `mlflow_browser.py` stops counting parents at 11-J."""
    anchors = sorted(
        str(path.relative_to(PKG)) for path in PKG.rglob("*.py")
        if "parents[" in path.read_text()
    )
    assert set(anchors) <= set(PARENTS_ANCHORS), (
        f"a new filesystem anchor appeared: {sorted(set(anchors) - set(PARENTS_ANCHORS))}"
    )
    assert "results/paths.py" in anchors


def test_mlflow_app_contract(mlflow_probe):
    """The MLflow browser: page registry, callbacks, layout and configuration."""
    observed = mlflow_probe["mlflow"]

    assert observed["module"] in (
        (MLFLOW_MODULES["pre"], MLFLOW_MODULES["post"]) if ALLOW_PRE_MLFLOW_LOCATIONS
        else (MLFLOW_MODULES["post"],))
    assert observed["page_file"] in (
        (MLFLOW_PAGE["pre"], MLFLOW_PAGE["post"]) if ALLOW_PRE_MLFLOW_LOCATIONS
        else (MLFLOW_PAGE["post"],))

    for field in ("index_status", "registry", "dependencies", "dependencies_digest",
                  "layout_digest", "page_globals"):
        assert observed[field] == MLFLOW[field], (
            f"{field} changed\n    expected {MLFLOW[field]!r}\n    observed {observed[field]!r}"
        )

    # The pages folder follows the package, so only its prefix may change.
    assert observed["config"]["use_pages"] is True
    assert observed["config"]["pages_folder"].endswith("pages")
    assert observed["config"]["assets_folder"].endswith("assets")


def test_mlruns_dir_equivalence():
    """Replacing `parents[4]` with `MLRUNS_DIR` must resolve to the same production directory."""
    page = None
    for candidate in (PKG / MLFLOW_PAGE["pre"], PKG / MLFLOW_PAGE["post"]):
        if candidate.is_file():
            page = candidate
    assert page is not None, "the MLflow browser page is missing"

    completed = subprocess.run(
        [sys.executable, "-c", "from noisyvis.results.paths import MLRUNS_DIR; print(MLRUNS_DIR)"],
        capture_output=True, text=True, timeout=120,
        env={key: value for key, value in __import__("os").environ.items()
             if key != "NOISYVIS_ROOT"},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    mlruns_dir = Path(completed.stdout.strip())

    assert page.resolve().parents[4] / "data" / "mlruns" == mlruns_dir, (
        "MLRUNS_DIR does not resolve to the directory the page's parents[4] anchor names"
    )
    assert mlruns_dir == WORKSPACE / "data" / "mlruns", mlruns_dir
    assert not (WORKSPACE / "src" / "data").exists(), (
        "src/data exists: an MLflow file store was created under the source tree (R18)"
    )
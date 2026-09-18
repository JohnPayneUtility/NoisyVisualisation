"""STN/LON data-store callbacks: the selected LON PID, the LON and STN run data for the current
selection, and the processed STN series that feed the shared-graph plot."""

import pandas as pd
from dash import Input, Output, State

from ..instance import app
from ..data import df, df_LONs, LON_display_columns
from ..helpers import _filter_penalty, convert_to_split_edges_format
from ..layout.stores import LON_TABLE_SELECTED_PID_STORE


# Store PID from selected LON row so it is accessible from the initial layout
@app.callback(
    Output(LON_TABLE_SELECTED_PID_STORE, 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data"),
    prevent_initial_call=True
)
def update_lon_table_selected_pid(selected_rows, lon_table_data):
    if selected_rows and lon_table_data:
        return lon_table_data[selected_rows[0]].get("PID")
    return None

# Filter LON dataframe by selected problem using table 1 selection
@app.callback(
    Output('LON_data', 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data")
)
def update_lon_data(selected_rows, LON_table_data):
    if not selected_rows:
        blank_df = pd.DataFrame(columns=df_LONs.columns)
        return blank_df.to_dict('records')

    selected_data = [LON_table_data[i] for i in selected_rows]

    # columns needed for plotting; feasibility may be absent for regular LONs
    LON_plotting_cols = [
        'local_optima', 'fitness_values', 'edges',
        'optima_feasibility', 'neighbour_feasibility', 'visit_proportions',
    ]

    df_filtered = pd.DataFrame(selected_data)
    mask = pd.Series(True, index=df_LONs.index)
    for col in LON_display_columns:
        if col in df_filtered.columns:
            allowed = df_filtered[col].unique()
            if any(pd.isnull(allowed)):
                mask &= (df_LONs[col].isin(allowed) | df_LONs[col].isnull())
            else:
                mask &= df_LONs[col].isin(allowed)

    df_result = df_LONs[mask]
    # tolerate missing feasibility columns
    df_result = df_result.loc[:, [c for c in LON_plotting_cols if c in df_result.columns]]
    rows = df_result.to_dict('records')

    combined = {
        "local_optima": [],
        "fitness_values": [],
        "edges": {},
        # feasibility maps (string-keyed for JSON)
        "opt_feas_map": {},      # "1,0,1,..." -> 0/1
        "neigh_feas_map": {},    # "1,0,1,..." -> float in [0,1]
        "visit_prop_map": {},    # "1,0,1,..." -> float (visit_proportion)
    }

    def key_str_from_opt(opt):
        # opt is a list/tuple of bits
        return ",".join(str(int(x)) for x in opt)

    for row in rows:
        los = row.get("local_optima", [])
        fvs = row.get("fitness_values", [])
        feas_raw = row.get("optima_feasibility")
        feas_list = feas_raw if isinstance(feas_raw, list) else [0] * len(los)
        neigh_raw = row.get("neighbour_feasibility")
        neigh_list = neigh_raw if isinstance(neigh_raw, list) else [0.0] * len(los)
        vp_raw = row.get("visit_proportions")
        vp_list = vp_raw if isinstance(vp_raw, list) else [0.0] * len(los)

        combined["local_optima"].extend(los)
        combined["fitness_values"].extend(fvs)

        # merge edges (weights) — keep tuple internally then convert in split-format helper
        for (source, target), weight in row.get("edges", {}).items():
            source = tuple(source)
            target = tuple(target)
            combined["edges"][(source, target)] = combined["edges"].get((source, target), 0) + weight

        # fill feasibility and visit maps (string keys for JSON safety)
        for opt, feas, neigh, vp in zip(los, feas_list, neigh_list, vp_list):
            k = key_str_from_opt(opt)
            combined["opt_feas_map"].setdefault(k, int(feas))
            combined["neigh_feas_map"].setdefault(k, float(neigh))
            combined["visit_prop_map"].setdefault(k, float(vp))

    # your helper expects only core keys; convert & then attach the maps
    payload_for_split = {
        "local_optima": combined["local_optima"],
        "fitness_values": combined["fitness_values"],
        "edges": combined["edges"],
    }
    dict_result_SE = convert_to_split_edges_format(payload_for_split)

    # attach JSON-safe feasibility and visit maps
    dict_result_SE["opt_feas_map"] = combined["opt_feas_map"]
    dict_result_SE["neigh_feas_map"] = combined["neigh_feas_map"]
    dict_result_SE["visit_prop_map"] = combined["visit_prop_map"]

    return dict_result_SE

@app.callback(
    Output('STN_data', 'data'),
    Input("table2", "selected_rows"),
    Input('penalty-filter-dropdown', 'value'),
    State("table2", "data")
)
def update_stn_data(selected_rows, penalty_value, table2_data):
    if not selected_rows:
        blank_df = pd.DataFrame(columns=df.columns)
        return blank_df.to_dict('records')

    # rows selected in table2 (deduped algorithm rows)
    selected_data = [table2_data[i] for i in selected_rows]
    df_selected = pd.DataFrame(selected_data)

    # Only filter by the identity keys for an algorithm series
    KEYS = ["PID", "fit_func", "algo_type", "algo_name", "noise", "experiment_name"]

    mask = pd.Series(True, index=df.index)
    for col in KEYS:
        # skip if missing for any reason
        if col not in df_selected.columns or col not in df.columns:
            continue
        col_vals = df_selected[col]
        has_nulls = col_vals.isna().any()
        allowed = col_vals.dropna().unique()
        if len(allowed) == 0 and has_nulls:
            mask &= df[col].isna()
        elif has_nulls:
            mask &= df[col].isin(allowed) | df[col].isna()
        else:
            mask &= df[col].isin(allowed)

    df_result = _filter_penalty(df[mask], penalty_value)

    print(
        f"[STN filter] selected_rows={selected_rows} -> matched_rows={len(df_result)}",
        flush=True
    )
    return df_result.to_dict('records')

@app.callback(
    [Output('STN_data_processed', 'data'),
     Output('STN_series_labels', 'data'),
     Output('noisy_fitnesses_data', 'data'),
     Output('STN_MO_data', 'data'),
     Output('STN_MO_series_labels', 'data'),
     Output('MO_data_PPP', 'data')],
    [Input('STN_data', 'data'),
     Input('mo_plot_type', 'value')],
)
def process_STN_data(df, mo_plot_type, group_cols=['algo_name', 'noise']):
    print('Processing data...', flush=True)
    df = pd.DataFrame(df)
    STN_data, STN_series, Noise_data = [], [], []
    MO_data, MO_series = [], []
    MO_data_PPP = []

    if df.empty:
        return STN_data, STN_series, Noise_data, MO_data, MO_series, MO_data_PPP

    # default/fallback if somehow empty
    mode = mo_plot_type or 'npnhv'

    grouped = df.groupby(group_cols)
    
    required = {'rep_sols','rep_fits','rep_noisy_fits','sol_iterations','sol_transitions'}
    has_required = required.issubset(df.columns)

    for group_key, group_df in grouped:
        runs = []

        for _, row in group_df.iterrows():
            if not has_required:
                continue
            if row['rep_sols'] is None:
                continue

            runs.append([
                row['rep_sols'],
                row['rep_fits'],
                row['rep_noisy_fits'],
                row['sol_iterations'],
                row['sol_transitions'],
                [],                                             # index 5: noisy_sol_variants removed
                row.get('rep_fitness_boxplot_stats', []),       # index 6: replaces noisy_variant_fitnesses
                row.get('rep_noisy_sols', []),
                row.get('rep_estimated_fits_whenadopted', []),
                row.get('rep_estimated_fits_whendiscarded', []),
                row.get('count_estimated_fits_whenadopted', []),
                row.get('count_estimated_fits_whendiscarded', []),
                row.get('sol_iterations_evals', []),      # index 12: evals-based iterations
                row.get('alternative_rep_sols', []),      # index 13: alt representation solutions
                row.get('alternative_rep_fits', []),      # index 14: alt representation fitnesses
            ])

        STN_data.append(runs)
        STN_series.append(group_key)

        # --- MO runs (mode-dependent) ---
        mo_runs = []
        mo_runs_full = []
        for _, row in group_df.iterrows():
            pareto_solutions = (row.get('pareto_solutions') or [])
            nps_hv_noisy         = (row.get('noisy_pf_noisy_hypervolumes') or [])
            nps_hv_true          = (row.get('noisy_pf_true_hypervolumes') or [])
            true_pareto_solutions = (row.get('true_pareto_solutions') or [])
            tps_hv_true         = (row.get('true_pf_hypervolumes') or [])
            nps_noisy_fits      = (row.get('pareto_fitnesses') or [])
            nps_clean_fits      = (row.get('pareto_true_fitnesses') or [])
            tps_clean_fits      = (row.get('true_pareto_fitnesses') or [])

            Gmax = min(len(pareto_solutions), len(nps_hv_noisy))

            fronts = []
            fronts_full = []
            for g in range(Gmax):
                if mode == 'bpbhv': # Both pareto front sets & both metrics
                    fronts.append({
                        'front1':   true_pareto_solutions[g],
                        'front2':   pareto_solutions[g],
                        'metric1':  tps_hv_true[g],
                        'metric2':  nps_hv_noisy[g],
                        'gen_idx':  g,
                    })
                elif mode == 'bpbhv_algo_pov': # Both pareto front sets & both metrics (algo POV)
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   true_pareto_solutions[g],
                        'metric1':  nps_hv_noisy[g],
                        'metric2':  tps_hv_true[g],
                        'gen_idx':  g,
                    })
                elif mode == 'tpthv':
                    fronts.append({
                        'front1':   true_pareto_solutions[g],
                        'front2':   None,
                        'metric1':  tps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                elif mode == 'npthv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                elif mode == 'npbhv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_true[g],
                        'metric2':  nps_hv_noisy[g],
                        'gen_idx':  g,
                    })
                elif mode == 'tpbhv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  tps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                else:  # npnhv
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_noisy[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                fronts_full.append({
                    'algo_front_solutions': pareto_solutions[g],
                    'algo_front_noisy_fitnesses': nps_noisy_fits[g],
                    'algo_front_clean_fitnesses': nps_clean_fits[g],
                    'algo_front_noisy_hypervolume': nps_hv_noisy[g],
                    'algo_front_clean_hypervolume': nps_hv_true[g],
                    'clean_front_solutions': true_pareto_solutions[g],
                    'clean_front_fitnesses': tps_clean_fits[g],
                    'clean_front_hypervolume': tps_hv_true[g],
                    'gen_idx':  g,
                })
            if fronts:
                mo_runs.append(fronts)
            if fronts_full:
                mo_runs_full.append(fronts_full)

        print(f'generations in data: {Gmax}', flush=True)
        MO_data.append(mo_runs)
        MO_series.append(group_key)
        MO_data_PPP.append(mo_runs_full)
    print('Data processing complete', flush=True)
    return STN_data, STN_series, Noise_data, MO_data, MO_series, MO_data_PPP

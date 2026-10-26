"""The real Dash callback chain of the Pareto Rank Layers controls (tests/test_pareto_rank_layers.py).

Run as a script, with cwd and NOISYVIS_ROOT a fresh temp root (harness.run_isolated.make_temp_root and
child_env) and the write fence installed. It writes a synthetic warehouse of real MO records (NSGA-II on the
knapsack, SEMO on COCZ, MoUMDA on the knapsack and a version-1 copy of the NSGA-II record) into the root,
imports the real dashboard application, and drives

    update_stn_data -> update_rank_layer_controls -> updateParetoPlot

over Dash's HTTP protocol (`/_dash-update-component`, as the browser makes the requests), starting from the
served layout's values and feeding each response into the next request. It asserts nothing: it prints one
JSON report, the events (what changed, what each callback returned, what the figure shows) and each run's
ground truth read independently from MORunView, for the test to check.

Nothing here runs at import time except when executed as a script.
"""

from __future__ import annotations

import collections
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KP_PID = "f1_l-d_kp_10_269"
# (series name, algorithm, problem, build parameters, mo_record version)
RUNS = [
    ("NSGA2", "NSGA2", "kp1", {"gen_limit": 40, "stop_without_improvement_in_gens": None}, 2),
    ("SEMO", "SEMO", "cocz1", {"gen_limit": 300, "stop_without_improvement_in_gens": None}, 2),
    ("MoUMDA", "MoUMDA", "kp1", {}, 2),
    ("NSGA2-v1", "NSGA2", "kp1", {"gen_limit": 40, "stop_without_improvement_in_gens": None}, 1),
]


def install_fence():
    spec = importlib.util.spec_from_file_location("_harness_fence", HERE / "fence.py")
    fence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fence)
    fence.install("/workspace")


def records():
    from harness.mo_runs import build, step_run
    from noisyvis.tracking.logger import clear_active_logger

    out = {}
    for series, name, prob, params, version in RUNS:
        clear_active_logger()
        algo = build(name, prob=prob, **params)
        step_run(algo)
        record = algo.eval_log.finalise()
        clear_active_logger()
        if version == 1:
            record = copy.deepcopy(record)
            record.pop("current_population")
            record["version"] = 1
        out[series] = record
    return out


def write_warehouse(root, mo_records):
    import numpy as np
    import pandas as pd

    base = dict(
        PID=KP_PID, problem_type="knapsack", problem_goal="maximise", problem_name="kp_10", dimensions=10,
        opt_global=30.0, fit_func="eval_noisy_kp_v1_mo", experiment_name="rank-layers",
        experiment_description="synthetic rank layers", algo_type="MO", noise=1.0, seed=1,
        stop_trigger="gen_limit", n_unique_sols=0, final_fit=np.nan, max_fit=np.nan, min_fit=np.nan,
        penalty=0.0, peak_ram_mb=120.0, rep_sols=None, rep_fits=np.nan, rep_noisy_fits=np.nan,
        rep_noisy_sols=None, rep_fitness_boxplot_stats=[], rep_estimated_fits_whenadopted=[],
        rep_estimated_fits_whendiscarded=[], count_estimated_fits_whenadopted=[],
        count_estimated_fits_whendiscarded=[], sol_iterations=None, sol_iterations_evals=None,
        sol_transitions=None, alternative_rep_sols=[], alternative_rep_fits=[],
    )
    rows = [dict(base, algo_name=series, seed_signature=f"sig-{series}",
                 n_gens=record["meta"]["n_generations"], n_evals=record["meta"]["n_evals"], mo_record=record)
            for series, record in mo_records.items()]
    lon = [dict(PID=KP_PID, problem_name="kp_10", problem_type="knapsack", problem_goal="maximise",
                dimensions=10, opt_global=30.0, fit_func="eval_noisy_kp_v1", algo_name="ILS", noise=0.0,
                n_runs=1, iterations=1, local_optima=[[0] * 10], fitness_values=[1.0], edges={},
                optima_feasibility=[1], neighbour_feasibility=[1.0], visit_counts=[1], visit_proportions=[1.0])]
    warehouse = root / "data" / "warehouse"
    pd.DataFrame(rows).to_pickle(warehouse / "algo_results.pkl")
    pd.DataFrame(lon).to_pickle(warehouse / "lon_results.pkl")


def truth(view):
    """A run's steps and ranks, read independently: the generations where the member multiset changes."""
    if not view.has_population:
        return {"version": view.version, "n_generations": view.n_generations}
    counters = [collections.Counter(view.population(g).eval_ids.tolist()) for g in range(view.n_generations)]
    starts = [g for g in range(view.n_generations) if g == 0 or counters[g] != counters[g - 1]]
    return {
        "version": view.version, "n_generations": view.n_generations, "starts": starts,
        "noisy_ranks": [int(view.population(g).noisy_rank.max()) + 1 for g in range(view.n_generations)],
        "true_ranks": [int(view.population(g).true_rank.max()) + 1 for g in range(view.n_generations)],
    }


def layout_values(node, values):
    """Every prop of every component with an id in the served layout, as '<id>.<prop>' -> value."""
    if isinstance(node, list):
        for child in node:
            layout_values(child, values)
    elif isinstance(node, dict):
        props = node.get("props")
        if isinstance(props, dict):
            if isinstance(props.get("id"), str):
                for prop, value in props.items():
                    values[f"{props['id']}.{prop}"] = value
            for value in props.values():
                layout_values(value, values)
    return values


def split(output):
    if output.startswith(".."):
        return [dict(zip(("id", "property"), part.split("."))) for part in output[2:-2].split("...")]
    component, prop = output.split(".")
    return {"id": component, "property": prop}


class Browser:
    """The browser's side of the protocol: current prop values, and one POST per callback call."""

    def __init__(self, app):
        self.client = app.server.test_client()
        self.specs = list(app._callback_list)
        self.values = layout_values(json.loads(self.client.get("/_dash-layout").get_data(as_text=True)), {})

    def spec(self, output_prop):
        [spec] = [s for s in self.specs if output_prop in s["output"].strip(".").split("...")
                  or s["output"] == output_prop]
        return spec

    def call(self, output_prop, changed):
        spec = self.spec(output_prop)
        body = {
            "output": spec["output"], "outputs": split(spec["output"]),
            "inputs": [{"id": i["id"], "property": i["property"], "value": self.values.get(f"{i['id']}.{i['property']}")}
                       for i in spec["inputs"]],
            "state": [{"id": s["id"], "property": s["property"], "value": self.values.get(f"{s['id']}.{s['property']}")}
                      for s in spec["state"]],
            "changedPropIds": list(changed),
        }
        response = self.client.post("/_dash-update-component", json=body)
        if response.status_code != 200:
            return {"status": response.status_code, "body": response.get_data(as_text=True)[-2000:]}
        payload = json.loads(response.get_data(as_text=True))
        returned = {}
        for component, props in payload.get("response", {}).items():
            for prop, value in props.items():
                returned[f"{component}.{prop}"] = value
                self.values[f"{component}.{prop}"] = value
        return {"status": 200, "returned": returned}


RANK_RE = re.compile(r"^Rank (\d+) ")
GEN_RE = re.compile(r", Gen (\d+)<br>")

CONTROL_KEYS = ("paretoRankRun.value", "paretoRankBasis.value", "paretoRankStep.value", "paretoRankStep.max", "paretoRankGeneration.value",
                "paretoRankGeneration.max", "paretoRankGeneration.disabled", "paretoRankStepLabel.children",
                "paretoRankMax.value", "paretoRankMax.max", "paretoRankMax.disabled")


def figure_summary(figure):
    title = (figure.get("layout", {}).get("title") or {}).get("text", "")
    gen = GEN_RE.search(title)
    ranks = sorted(int(m.group(1)) for t in figure.get("data", [])
                   if (m := RANK_RE.match(t.get("name") or "")))
    marker_order = [int(m.group(1)) for t in figure.get("data", []) if (m := RANK_RE.match(t.get("name") or ""))]
    members = sum(len(t.get("x") or []) for t in figure.get("data", []) if RANK_RE.match(t.get("name") or ""))
    return {"title": title, "gen": int(gen.group(1)) if gen else None, "ranks": ranks,
            "marker_order": marker_order, "points": members, "n_traces": len(figure.get("data", []))}


def main():
    install_fence()
    sys.path.insert(0, str(HERE.parent))
    root = Path(os.environ["NOISYVIS_ROOT"])
    mo_records = records()
    write_warehouse(root, mo_records)

    from noisyvis.dashboard.app import app
    from noisyvis.dashboard.data import df
    from noisyvis.results.mo_view import MORunView

    row_of = {series: int(df.index[df["algo_name"] == series][0]) for series in mo_records}
    report = {"rows": row_of, "truth": {series: truth(MORunView(mo_records[series])) for series in mo_records},
              "events": []}
    browser = Browser(app)
    table2 = [{"PID": KP_PID, "fit_func": "eval_noisy_kp_v1_mo", "algo_type": "MO", "algo_name": series,
               "noise": 1.0, "experiment_name": "rank-layers"} for series in mo_records]

    def event(name, changed=(), select=None, set_values=None, controls=True, plot=True, plot_type="RankLayers"):
        """One user action: set props (as the browser would), run the callbacks it triggers, record them."""
        record = {"name": name}
        for key, value in (set_values or {}).items():
            browser.values[key] = value
        browser.values["paretoFrontPlotType.value"] = plot_type
        if select is not None:
            browser.values["table2.data"] = [table2[i] for i in select]
            browser.values["table2.selected_rows"] = list(range(len(select)))
            browser.values["penalty-filter-dropdown.value"] = None
            stn = browser.call("STN_data.data", ["table2.selected_rows"])
            record["stn_status"] = stn["status"]
            changed = ["STN_data.data", *changed]
        if controls:
            controls = browser.call("paretoRankStepLabel.children", changed)
            record["controls_status"] = controls["status"]
            record["controls_body"] = controls.get("body")
            record["returned"] = sorted(controls.get("returned", {}))
        record["controls"] = {key: browser.values.get(key) for key in CONTROL_KEYS}
        record["run_options"] = browser.values.get("paretoRankRun.options")
        if plot:
            figure = browser.call("plotParetoFront.figure", list(changed) or ["paretoFrontPlotType.value"])
            record["plot_status"] = figure["status"]
            record["plot_body"] = figure.get("body")
            record["figure"] = figure_summary(browser.values.get("plotParetoFront.figure", {}))
        report["events"].append(record)
        return record

    toggled = browser.call("paretoRankLayerOptions.style", ["paretoFrontPlotType.value"])
    report["layout_defaults"] = {key: browser.values.get(key) for key in CONTROL_KEYS}

    browser.values["paretoFrontPlotType.value"] = "RankLayers"
    report["toggle"] = {"default": toggled.get("returned"),
                        "rank_layers": browser.call("paretoRankLayerOptions.style",
                                                    ["paretoFrontPlotType.value"]).get("returned")}
    browser.values["paretoFrontPlotType.value"] = "Basic"
    report["toggle"]["basic"] = browser.call("paretoRankLayerOptions.style", ["paretoFrontPlotType.value"]).get("returned")

    # the selection's runs are listed by series name, so MoUMDA's is the first (the default) run
    first, semo = report["truth"]["MoUMDA"], report["truth"]["SEMO"]
    event("initial")                                         # the page load: no selection yet
    event("select", select=[0, 1, 2, 3])
    event("step", ["paretoRankStep.value"], set_values={"paretoRankStep.value": 3})
    event("generation", ["paretoRankGeneration.value"], set_values={"paretoRankGeneration.value": 7})
    event("generation_out_of_range", ["paretoRankGeneration.value"],
          set_values={"paretoRankGeneration.value": 10 ** 6})
    event("generation_negative", ["paretoRankGeneration.value"], set_values={"paretoRankGeneration.value": -4})
    event("final", ["paretoRankLast.n_clicks"], set_values={"paretoRankLast.n_clicks": 1})
    event("first", ["paretoRankFirst.n_clicks"], set_values={"paretoRankFirst.n_clicks": 1})

    # a generation with at least three noisy ranks, then an explicit highest rank and a basis change
    deep = next((g for g, n in enumerate(first["noisy_ranks"]) if n >= 3 and g), None)
    report["deep_generation"] = deep
    if deep is not None:
        event("deep_generation", ["paretoRankGeneration.value"], set_values={"paretoRankGeneration.value": deep})
        browser.values["paretoRankMax.value"] = 1                     # the rank slider feeds the plot only
        event("rank_max_1", ["paretoRankMax.value"], controls=False)
        event("basis_true", ["paretoRankBasis.value"], set_values={"paretoRankBasis.value": "true"})
        event("basis_noisy", ["paretoRankBasis.value"], set_values={"paretoRankBasis.value": "noisy"})
        browser.values["paretoRankMax.value"] = browser.values["paretoRankMax.max"]
        event("rank_max_all", ["paretoRankMax.value"], controls=False)

    event("run_semo", ["paretoRankRun.value"], set_values={"paretoRankRun.value": row_of["SEMO"]})
    # a generation inside a SEMO step that spans several generations, but not its first one
    long_steps = [(s, e) for s, e in zip(semo["starts"], semo["starts"][1:] + [semo["n_generations"]]) if e - s > 2]
    mid = long_steps[0][0] + 1 if long_steps else None
    report["semo_mid_generation"] = mid
    if mid is not None:
        event("semo_mid_generation", ["paretoRankGeneration.value"], set_values={"paretoRankGeneration.value": mid})
    # several inputs changed in one request
    event("run_and_generation", ["paretoRankRun.value", "paretoRankGeneration.value"],
          set_values={"paretoRankRun.value": row_of["NSGA2"], "paretoRankGeneration.value": 5})
    event("step_and_generation", ["paretoRankStep.value", "paretoRankGeneration.value"],
          set_values={"paretoRankStep.value": 1, "paretoRankGeneration.value": 9})
    event("basis_and_step", ["paretoRankBasis.value", "paretoRankStep.value"],
          set_values={"paretoRankBasis.value": "noisy", "paretoRankStep.value": 2})
    # the selected run leaves the selection: the first remaining run, at its final population
    event("deselect_nsga2", select=[2, 3, 1])
    event("run_v1", ["paretoRankRun.value"], set_values={"paretoRankRun.value": row_of["NSGA2-v1"]})
    event("v1_only", select=[3])
    event("empty_selection", select=[])
    # the existing plot types still render through the same app, ignoring the rank-layer controls
    event("basic_plot", select=[0, 1], plot_type="Basic")
    report["basic_trace_counts"] = {}
    for plot_type in ("Basic", "Subplots", "paretoAnimation", "NoisyPopulationVsClean", "IndVsDist", "Scatter"):
        browser.values["paretoFrontPlotType.value"] = plot_type
        figure = browser.call("plotParetoFront.figure", ["paretoFrontPlotType.value"])
        data = browser.values.get("plotParetoFront.figure", {}).get("data", [])
        report["basic_trace_counts"][plot_type] = [figure["status"], len(data)]

    print(json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

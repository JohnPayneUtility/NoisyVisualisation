"""The persistent MO run record (MO recording plan v6, Work Group 5): finalise(), schema and validation.

    finalise()      freezes an MOEvaluationLogger into a plain-data mo_record (NumPy arrays and Python
                    primitives only), once per run, with no evaluation, RNG or change to the log;
                    repeated calls give equal records
    purity          no DEAP, logger, dataclass or noisyvis object anywhere: the record unpickles in a
                    fresh interpreter that imports neither deap nor noisyvis
    unknown f       a genotype only ever seen as x~ has no recorded f (NaN row, first_orig_eval -1):
                    f(x~) is never inferred from an event whose true objectives are f(x)
    validation      MORunView rejects malformed or incompatible records at construction, including
                    canonical observation identity checked independently of finalise()

Runs in-process (plus one clean subprocess) and writes nothing.
"""

from __future__ import annotations

import copy
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest

from harness.mo_runs import A, B, C, F_A, F_B, F_C, Lab, build, rng_state, step_run
from noisyvis.common import pareto
from noisyvis.results import mo_view
from noisyvis.results.mo_view import MORecordError, MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.tracking.mo_logger import EvaluationLogError


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


def finalised(name="MoUMDA", prob="kp1"):
    algo = build(name, prob=prob)
    step_run(algo)
    return algo, algo.eval_log.finalise()


def records_equal(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() \
            and all(records_equal(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype \
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return type(a) is type(b) and a == b


PLAIN = (str, int, float, bool, type(None))


def assert_plain(value, where="mo_record"):
    """Only dicts, tuples, primitives and numeric NumPy arrays, recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), f"{where}: non-string key {key!r}"
            assert_plain(item, f"{where}.{key}")
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            assert_plain(item, f"{where}[{i}]")
    elif isinstance(value, np.ndarray):
        assert value.dtype.kind in "biuf", f"{where}: array dtype {value.dtype}"
    else:
        assert type(value) in PLAIN, f"{where}: {type(value).__name__} is not plain data"


# ------------------------------------------------------------------------------ finalise

@pytest.mark.parametrize("prob", ["kp0", "kp1", "prior"])
@pytest.mark.parametrize("name", ["SEMO", "NSGA2", "MoUMDA", "MoUMDA_ParetoArchive", "MoUMDA_KMeans"])
def test_record_schema_types_and_purity(name, prob):
    algo, record = finalised(name, prob)
    log = algo.eval_log
    assert_plain(record)
    assert (record["schema"], record["version"]) == (mo_view.MO_RECORD_SCHEMA, mo_view.MO_RECORD_VERSION)
    meta = record["meta"]
    assert (meta["n_evals"], meta["n_generations"]) == (algo.evals, algo.gens + 1) == (len(log), log.n_generations)
    assert (meta["n_genotypes"], meta["n_observations"], meta["n_obj"], meta["sol_length"]) == \
        (len(log.genotypes), len(log.obs_first_eval), 2, 10)
    assert meta["opt_weights"] == algo.opt_weights and meta["ref_point"] == tuple(algo.ref_point)
    assert meta["gene_dtype"] == "uint8" and record["genotypes"]["values"].dtype == np.uint8
    events = record["events"]
    assert events["orig_geno"].dtype == events["obs_id"].dtype == np.int32 and events["obs_obj"].dtype == np.float64
    assert meta["prior_noise_observed"] is (prob == "prior")
    assert (events["eval_geno"] is None) is (prob != "prior")
    for block in ("noisy_archive", "clean_archive"):
        assert all(record[block][k].dtype == np.int32 for k in ("member", "enter_eval", "exit_eval", "exit_by"))
    MORunView(record)  # validates


def test_record_unpickles_without_deap_or_noisyvis():
    _, record = finalised("NSGA2", "prior")
    script = ("import pickle, sys\n"
              "record = pickle.loads(sys.stdin.buffer.read())\n"
              "assert record['schema'] == 'noisyvis.mo_record'\n"
              "print(sorted(m for m in sys.modules if m.split('.')[0] in ('deap', 'noisyvis')))\n")
    done = subprocess.run([sys.executable, "-c", script], input=pickle.dumps(record), capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode()[-2000:]
    assert done.stdout.decode().strip() == "[]"


def test_finalise_is_idempotent_and_observational():
    algo, _ = finalised("MoUMDA_ParetoArchive", "prior")
    log = algo.eval_log
    columns = repr([log.orig_geno, log.eval_geno, log.true_objectives, log.observed, log.obs_id, log.gen_last_eval,
                    log.noisy_front_members, log.clean_front_members, log.genotypes])
    random.seed(9)
    np.random.seed(9)
    before = rng_state()
    first, second = log.finalise(), log.finalise()
    assert rng_state() == before
    assert records_equal(first, second) and first is not second
    assert first["events"]["obs_obj"] is not second["events"]["obs_obj"]
    assert repr([log.orig_geno, log.eval_geno, log.true_objectives, log.observed, log.obs_id, log.gen_last_eval,
                 log.noisy_front_members, log.clean_front_members, log.genotypes]) == columns


def test_finalise_requires_a_generation_boundary():
    lab = Lab()
    with pytest.raises(EvaluationLogError, match="generation boundary"):
        lab.log.finalise()                                       # nothing observed
    a = lab.evaluate(A, F_A, (80.0, 70.0))
    lab.observe([a])
    lab.log.finalise()
    lab.evaluate(B, F_B, (95.0, 55.0))                           # an event after the last boundary
    with pytest.raises(EvaluationLogError, match="generation boundary"):
        lab.log.finalise()

    def finalising(individual):
        lab.log.finalise()
        return (1.0, 1.0)

    with pytest.raises(EvaluationLogError, match="in progress"):
        lab.log.evaluate(finalising, list(C), {})


def test_genotypes_seen_only_as_x_tilde_have_no_recorded_true_objectives():
    """Eval 20 evaluates B as x~ of C; eval 80 generates B. B is interned at 20, yet f(B) is recorded
    only from eval 80 (first_orig_eval 80), never inferred from eval 20 (whose f is f(C)). A genotype that
    is never generated keeps a NaN row and first_orig_eval -1."""
    lab = Lab()
    filler = [0, 1, 1, 1]
    for _ in range(20):
        lab.evaluate(filler, (1, 99), (1.0, 99.0))                                # events 0..19
    lab.evaluate(C, F_C, (10.0, 10.0), evaluated=B)                               # event 20: B only as x~
    lab.evaluate(C, F_C, (10.0, 10.0), evaluated=[1, 1, 1, 1])                    # event 21: never generated
    for _ in range(58):
        lab.evaluate(filler, (1, 99), (1.0, 99.0))                                # events 22..79
    b = lab.evaluate(B, (90, 10), (90.0, 10.0))                                   # event 80: B as x
    lab.observe([b])
    record = lab.log.finalise()
    genotypes = record["genotypes"]
    g_b, g_never = lab.geno(B), lab.geno([1, 1, 1, 1])
    assert g_b < g_never and record["events"]["eval_geno"][20] == g_b
    assert genotypes["first_orig_eval"][g_b] == 80 and tuple(genotypes["true_obj"][g_b]) == (90.0, 10.0)
    assert genotypes["first_orig_eval"][g_never] == -1 and np.isnan(genotypes["true_obj"][g_never]).all()
    view = MORunView(record)
    assert tuple(view.true_objectives(g_b)) == (90.0, 10.0)
    with pytest.raises(KeyError, match="never generated"):
        view.true_objectives(g_never)
    assert view.clean_archive_history().starts[-1] == 80


def test_real_prior_runs_record_unknown_true_objectives_explicitly():
    _, record = finalised("MoUMDA", "prior")
    genotypes = record["genotypes"]
    never = genotypes["first_orig_eval"] < 0
    assert never.any(), "prior noise evaluates genotypes the optimiser never generated"
    assert np.isnan(genotypes["true_obj"][never]).all() and np.isfinite(genotypes["true_obj"][~never]).all()


def test_schema_constants_agree_with_the_replay():
    assert mo_view.NOT_EXITED == pareto.NOT_EXITED


# ------------------------------------------------------------------------------ validation

@pytest.fixture(scope="module")
def twin_record():
    """A zero-noise MoUMDA record (it has exact (x, x~, y) twins) and its prior-noise counterpart."""
    return finalised("MoUMDA", "kp0")[1], finalised("MoUMDA", "prior")[1]


def twins(record):
    obs = record["events"]["obs_id"]
    later = [e for e in range(len(obs)) if obs[e] in obs[:e]]
    assert later, "the zero-noise run has twins"
    return later[0], int(np.flatnonzero(obs == obs[later[0]])[0])


def corrupt_same_obs_different_y(record):
    e, _ = twins(record)
    record["events"]["obs_obj"][e, 0] += 1.0


def corrupt_same_key_two_obs(record):
    """A twin after the last new observation gets a fresh obs_id: the numbering stays first-appearance
    ordered, but one (x, x~, y) now has two obs_ids."""
    obs = record["events"]["obs_id"]
    _, first_seen = np.unique(obs, return_index=True)
    late_twins = [e for e in range(int(first_seen.max()) + 1, len(obs))]
    assert late_twins, "the zero-noise run ends with re-evaluated twins"
    obs[late_twins[0]] = record["meta"]["n_observations"]
    record["meta"]["n_observations"] += 1


def corrupt_numbering(record):
    obs = record["events"]["obs_id"]
    a, b = obs == 0, obs == 1
    obs[a], obs[b] = 1, 0


CORRUPTIONS = {
    "schema": (lambda r: r.update(schema="other.record"), "schema"),
    "version": (lambda r: r.update(version=2), "version 2"),
    "missing block": (lambda r: r.pop("noisy_archive"), "missing"),
    "meta type": (lambda r: r["meta"].update(n_evals=float(r["meta"]["n_evals"])), "meta.n_evals"),
    "obs_id dtype": (lambda r: r["events"].update(obs_id=r["events"]["obs_id"].astype(float)), "dtype"),
    "event shape": (lambda r: r["events"].update(obs_obj=r["events"]["obs_obj"][:-1]), "shapes"),
    "geno range": (lambda r: r["events"]["orig_geno"].__setitem__(0, r["meta"]["n_genotypes"]), "outside"),
    "first_orig_eval": (lambda r: r["genotypes"]["first_orig_eval"].__setitem__(r["events"]["orig_geno"][-1], 0),
                        "first_orig_eval|finite"),
    "unknown f finite": (lambda r: r["genotypes"]["true_obj"].__setitem__(0, np.nan), "finite"),
    "prior flag": (lambda r: r["meta"].update(prior_noise_observed=True), "prior_noise_observed"),
    "boundaries": (lambda r: r["gen_last_eval"].__setitem__(-1, r["gen_last_eval"][-1] - 1), "gen_last_eval"),
    "front gens": (lambda r: r["current_noisy_front"]["gen"].__setitem__(0, 1), "current_noisy_front.gen"),
    "csr": (lambda r: r["current_clean_front"]["member_offsets"].__setitem__(-1, 10**6), "CSR"),
    "exit before enter": (lambda r: (r["noisy_archive"]["exit_eval"].__setitem__(0, r["noisy_archive"]["enter_eval"][0]),
                                     r["noisy_archive"]["exit_by"].__setitem__(0, 0)), "exit_eval"),
    "exit_by": (lambda r: r["clean_archive"]["exit_by"].__setitem__(
        int(np.flatnonzero(r["clean_archive"]["exit_eval"] == -1)[0]), 0), "exit_by"),
    "hv rows": (lambda r: r["clean_archive"].update(hv_clean=r["clean_archive"]["hv_clean"][:-1]), "one value"),
    "same obs, different y": (corrupt_same_obs_different_y, "identify exactly"),
    "same key, two obs": (corrupt_same_key_two_obs, "identify exactly"),
    "obs numbering": (corrupt_numbering, "first-appearance order"),
}


@pytest.mark.parametrize("case", sorted(CORRUPTIONS))
def test_malformed_records_are_rejected(twin_record, case):
    mutate, match = CORRUPTIONS[case]
    record = copy.deepcopy(twin_record[0])
    MORunView(record)
    mutate(record)
    with pytest.raises(MORecordError, match=match):
        MORunView(record)


def test_a_mapping_is_required_and_prior_records_validate(twin_record):
    with pytest.raises(MORecordError, match="mapping"):
        MORunView([1, 2, 3])
    MORunView(twin_record[1])
    broken = copy.deepcopy(twin_record[1])
    broken["events"]["eval_geno"] = None      # prior noise observed, but x~ dropped
    with pytest.raises(MORecordError, match="prior_noise_observed"):
        MORunView(broken)

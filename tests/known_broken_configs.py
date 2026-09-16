"""Configs that are known to be broken, and are deliberately not fixed in this migration.

Entries here become xfails in `test_config_resolution.py`. The marks are strict, so:

* an unlisted failure fails the suite;
* a listed failure is reported as xfail;
* a listed config that starts passing is an XPASS, which also fails -- the allowlist cannot
  silently go stale.

Adding an entry is a decision to defer a behavioural fix, not a way to quiet a failure caused by
the restructuring. Keys are paths relative to `configs/`.
"""

KNOWN_BROKEN = {
    "Multiobjective/MO_knapsack_test/mo_1p1ea.yaml":
        "B1: targets MOAlgorithms.MoMuPlusLamdaEA, which does not exist. Behavioural fix deferred.",
}

"""noisyvis.experiments — experiment orchestration behind the root run scripts (plan Stage 7).

Deliberately empty of imports and re-exports: every entry point imports this package, and any
import here would change module-load order and so the RNG call order the reproducibility baselines
pin (plan §7.1, R1). Import submodules explicitly.
"""

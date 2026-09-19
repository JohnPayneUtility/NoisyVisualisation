# docker/

## Stage 12: config dotted paths now name `noisyvis.*` modules

Stage 12 of the reorganisation rewrote every Hydra config dotted path from the temporary
compatibility modules (`src.algorithms.*`, `src.problems.*`, root `run_helpers`) to the canonical
`noisyvis.*` modules, and deleted those compatibility modules. Runs recorded from then on differ in
two places:

- New CoLON runs log the MLflow parameter `violation_fn` as
  `noisyvis.problems.constraints.knap_violation` (previously
  `src.problems.ViolationFunctions.knap_violation`).
- New `.hydra/config.yaml` artifacts contain the canonical `noisyvis.*` target paths.

Historical runs and artifacts keep the strings they were recorded with; they are not rewritten.

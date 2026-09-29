"""noisyvis.tracking — per-run evaluation tracking (plan Stage 8).

    logger  ExperimentLogger, its fit-history backends, and the module-level active-logger singleton
            (set_active_logger / get_active_logger / clear_active_logger) that fitness functions use
    mo_logger  MOEvaluationLogger: the run-scoped multi-objective evaluation log, _eval_tag provenance
            the current-front change-point histories and the passive historical archives;
            it uses the singleton above for the duration of each evaluation and holds no global state

Deliberately empty of imports and re-exports: the singleton state lives only in
`noisyvis.tracking.logger`, and it must stay the one module that holds it (risk R3). Import
`noisyvis.tracking.logger` explicitly.
"""

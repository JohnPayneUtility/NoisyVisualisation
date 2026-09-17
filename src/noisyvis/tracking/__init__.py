"""noisyvis.tracking — per-run evaluation tracking (plan Stage 8).

    logger  ExperimentLogger, its fit-history backends, and the module-level active-logger singleton
            (set_active_logger / get_active_logger / clear_active_logger) that fitness functions use

Deliberately empty of imports and re-exports: the singleton state lives only in
`noisyvis.tracking.logger`, and it must stay the one module that holds it (risk R3). Import
`noisyvis.tracking.logger` explicitly.
"""

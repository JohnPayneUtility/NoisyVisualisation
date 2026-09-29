"""Result persistence and queries (plan §5.2): the warehouse reader and writer together.

    paths         filesystem locations, anchored to the project root (§5.9)
    store         warehouse write (save_or_append_results) and read (load_*_results)
    mlflow_query  MLflow experiment and run listings for the MLflow browser app
    mo_view       MORunView: the NumPy-only read API over one persisted multi-objective run record
                  (mo_record), with its schema contract

Deliberately empty of imports, so `noisyvis.results.paths` stays importable without pandas or MLflow.
"""

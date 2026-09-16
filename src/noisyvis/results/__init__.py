"""Result persistence and queries (plan §5.2): the warehouse reader and writer together.

    paths         filesystem locations, anchored to the project root (§5.9)
    store         warehouse write (save_or_append_results) and read (load_*_results)
    mlflow_query  MLflow experiment and run listings for the MLflow browser app

Deliberately empty of imports, so `noisyvis.results.paths` stays importable without pandas or MLflow.
"""

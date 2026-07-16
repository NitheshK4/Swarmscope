from sandbox.storage.base import BaseStore
from sandbox.storage.duckdb_store import DuckDBStore

def get_store(db_path: str = "simulation_runs.duckdb") -> BaseStore:
    return DuckDBStore(db_path=db_path)

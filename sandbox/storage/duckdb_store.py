import duckdb
import json
from sandbox.storage.base import BaseStore
from sandbox.schemas import SimulationRun, SimulationMetadata, Message

class DuckDBStore(BaseStore):
    def __init__(self, db_path: str = "simulation_runs.duckdb"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = duckdb.connect(self.db_path)
        try:
            # Create runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    simulation_id VARCHAR PRIMARY KEY,
                    scenario_name VARCHAR,
                    timestamp VARCHAR,
                    total_turns INTEGER,
                    backend VARCHAR,
                    temperature DOUBLE,
                    status VARCHAR,
                    loop_score DOUBLE,
                    deadlock_score DOUBLE,
                    collusion_score DOUBLE,
                    goal_drift_score DOUBLE,
                    loop_explanation VARCHAR,
                    deadlock_explanation VARCHAR,
                    collusion_explanation VARCHAR,
                    goal_drift_explanation VARCHAR
                )
            """)
            # Create messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    simulation_id VARCHAR,
                    sender VARCHAR,
                    receiver VARCHAR,
                    content VARCHAR,
                    turn INTEGER,
                    timestamp VARCHAR
                )
            """)
        finally:
            conn.close()

    def save_run(self, run: SimulationRun) -> None:
        conn = duckdb.connect(self.db_path)
        try:
            m = run.metadata
            d_scores = run.detector_scores
            d_explanations = run.detector_explanations
            
            # Save run metadata and detector scores
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    simulation_id, scenario_name, timestamp, total_turns, backend, temperature, status,
                    loop_score, deadlock_score, collusion_score, goal_drift_score,
                    loop_explanation, deadlock_explanation, collusion_explanation, goal_drift_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.simulation_id, m.scenario_name, m.timestamp, m.total_turns, m.backend, m.temperature, m.status,
                d_scores.get("loop", 0.0), d_scores.get("deadlock", 0.0), d_scores.get("collusion", 0.0), d_scores.get("goal_drift", 0.0),
                d_explanations.get("loop", ""), d_explanations.get("deadlock", ""), d_explanations.get("collusion", ""), d_explanations.get("goal_drift", "")
            ))
            
            # Delete old messages for this simulation if any (for idempotency)
            conn.execute("DELETE FROM messages WHERE simulation_id = ?", (m.simulation_id,))
            
            # Insert messages
            for msg in run.messages:
                conn.execute("""
                    INSERT INTO messages (simulation_id, sender, receiver, content, turn, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    m.simulation_id, msg.sender, msg.receiver, msg.content, msg.turn, msg.timestamp
                ))
        finally:
            conn.close()

    def get_run(self, simulation_id: str) -> SimulationRun:
        conn = duckdb.connect(self.db_path)
        try:
            row = conn.execute("SELECT * FROM runs WHERE simulation_id = ?", (simulation_id,)).fetchone()
            if not row:
                raise ValueError(f"Simulation run '{simulation_id}' not found.")
            
            # Map columns
            # schema order: simulation_id, scenario_name, timestamp, total_turns, backend, temperature, status,
            # loop_score, deadlock_score, collusion_score, goal_drift_score,
            # loop_explanation, deadlock_explanation, collusion_explanation, goal_drift_explanation
            metadata = SimulationMetadata(
                simulation_id=row[0],
                scenario_name=row[1],
                timestamp=row[2],
                total_turns=row[3],
                backend=row[4],
                temperature=row[5],
                status=row[6]
            )
            
            scores = {
                "loop": row[7],
                "deadlock": row[8],
                "collusion": row[9],
                "goal_drift": row[10]
            }
            
            explanations = {
                "loop": row[11],
                "deadlock": row[12],
                "collusion": row[13],
                "goal_drift": row[14]
            }
            
            messages = self.get_run_messages(simulation_id)
            
            return SimulationRun(
                metadata=metadata,
                messages=messages,
                detector_scores=scores,
                detector_explanations=explanations
            )
        finally:
            conn.close()

    def get_all_runs(self) -> list[SimulationMetadata]:
        conn = duckdb.connect(self.db_path)
        try:
            rows = conn.execute("SELECT simulation_id, scenario_name, timestamp, total_turns, backend, temperature, status FROM runs ORDER BY timestamp DESC").fetchall()
            runs = []
            for r in rows:
                runs.append(SimulationMetadata(
                    simulation_id=r[0],
                    scenario_name=r[1],
                    timestamp=r[2],
                    total_turns=r[3],
                    backend=r[4],
                    temperature=r[5],
                    status=r[6]
                ))
            return runs
        finally:
            conn.close()

    def get_run_messages(self, simulation_id: str) -> list[Message]:
        conn = duckdb.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT sender, receiver, content, turn, timestamp 
                FROM messages 
                WHERE simulation_id = ? 
                ORDER BY turn ASC
            """, (simulation_id,)).fetchall()
            
            messages = []
            for r in rows:
                messages.append(Message(
                    sender=r[0],
                    receiver=r[1],
                    content=r[2],
                    turn=r[3],
                    timestamp=r[4]
                ))
            return messages
        finally:
            conn.close()

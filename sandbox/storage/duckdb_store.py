import duckdb
from sandbox.storage.base import BaseStore
from sandbox.schemas import SimulationRun, SimulationMetadata, Message

class DuckDBStore(BaseStore):
    def __init__(self, db_path: str = "simulation_runs.duckdb"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = duckdb.connect(self.db_path)
        try:
            # Create runs table with all detector score columns
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
                    jailbreak_score DOUBLE,
                    escalation_score DOUBLE,
                    information_leakage_score DOUBLE,
                    loop_explanation VARCHAR,
                    deadlock_explanation VARCHAR,
                    collusion_explanation VARCHAR,
                    goal_drift_explanation VARCHAR,
                    jailbreak_explanation VARCHAR,
                    escalation_explanation VARCHAR,
                    information_leakage_explanation VARCHAR
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

            # Migration: add new columns if they don't exist (for existing databases)
            existing_columns = set()
            try:
                cols = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'").fetchall()
                existing_columns = {c[0] for c in cols}
            except Exception:
                pass

            new_columns = {
                "jailbreak_score": "DOUBLE DEFAULT 0.0",
                "escalation_score": "DOUBLE DEFAULT 0.0",
                "information_leakage_score": "DOUBLE DEFAULT 0.0",
                "jailbreak_explanation": "VARCHAR DEFAULT ''",
                "escalation_explanation": "VARCHAR DEFAULT ''",
                "information_leakage_explanation": "VARCHAR DEFAULT ''"
            }
            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    try:
                        conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass  # Column may already exist
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
                    jailbreak_score, escalation_score, information_leakage_score,
                    loop_explanation, deadlock_explanation, collusion_explanation, goal_drift_explanation,
                    jailbreak_explanation, escalation_explanation, information_leakage_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.simulation_id, m.scenario_name, m.timestamp, m.total_turns, m.backend, m.temperature, m.status,
                d_scores.get("loop", 0.0), d_scores.get("deadlock", 0.0),
                d_scores.get("collusion", 0.0), d_scores.get("goal_drift", d_scores.get("goaldrift", 0.0)),
                d_scores.get("jailbreak", 0.0), d_scores.get("escalation", 0.0),
                d_scores.get("information_leakage", d_scores.get("informationleakage", 0.0)),
                d_explanations.get("loop", ""), d_explanations.get("deadlock", ""),
                d_explanations.get("collusion", ""), d_explanations.get("goal_drift", d_explanations.get("goaldrift", "")),
                d_explanations.get("jailbreak", ""), d_explanations.get("escalation", ""),
                d_explanations.get("information_leakage", d_explanations.get("informationleakage", ""))
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
            
            # Get column names for dynamic mapping
            col_names = [desc[0] for desc in conn.execute("SELECT * FROM runs LIMIT 0").description]
            row_dict = dict(zip(col_names, row))
            
            metadata = SimulationMetadata(
                simulation_id=row_dict["simulation_id"],
                scenario_name=row_dict["scenario_name"],
                timestamp=row_dict["timestamp"],
                total_turns=row_dict["total_turns"],
                backend=row_dict["backend"],
                temperature=row_dict["temperature"],
                status=row_dict["status"]
            )
            
            scores = {
                "loop": row_dict.get("loop_score", 0.0),
                "deadlock": row_dict.get("deadlock_score", 0.0),
                "collusion": row_dict.get("collusion_score", 0.0),
                "goal_drift": row_dict.get("goal_drift_score", 0.0),
                "jailbreak": row_dict.get("jailbreak_score", 0.0),
                "escalation": row_dict.get("escalation_score", 0.0),
                "information_leakage": row_dict.get("information_leakage_score", 0.0)
            }
            
            explanations = {
                "loop": row_dict.get("loop_explanation", ""),
                "deadlock": row_dict.get("deadlock_explanation", ""),
                "collusion": row_dict.get("collusion_explanation", ""),
                "goal_drift": row_dict.get("goal_drift_explanation", ""),
                "jailbreak": row_dict.get("jailbreak_explanation", ""),
                "escalation": row_dict.get("escalation_explanation", ""),
                "information_leakage": row_dict.get("information_leakage_explanation", "")
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
            rows = conn.execute("""
                SELECT simulation_id, scenario_name, timestamp, total_turns, backend, temperature, status,
                       loop_score, deadlock_score, collusion_score, goal_drift_score,
                       jailbreak_score, escalation_score, information_leakage_score
                FROM runs ORDER BY timestamp DESC
            """).fetchall()
            runs = []
            for r in rows:
                runs.append(SimulationMetadata(
                    simulation_id=r[0],
                    scenario_name=r[1],
                    timestamp=r[2],
                    total_turns=r[3],
                    backend=r[4],
                    temperature=r[5],
                    status=r[6],
                    loop_score=r[7] or 0.0,
                    deadlock_score=r[8] or 0.0,
                    collusion_score=r[9] or 0.0,
                    goal_drift_score=r[10] or 0.0,
                    jailbreak_score=r[11] or 0.0,
                    escalation_score=r[12] or 0.0,
                    information_leakage_score=r[13] or 0.0
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

    def delete_run(self, simulation_id: str) -> None:
        """Deletes a simulation run and its associated messages."""
        conn = duckdb.connect(self.db_path)
        try:
            # Check if run exists
            row = conn.execute("SELECT simulation_id FROM runs WHERE simulation_id = ?", (simulation_id,)).fetchone()
            if not row:
                raise ValueError(f"Simulation run '{simulation_id}' not found.")
            
            conn.execute("DELETE FROM messages WHERE simulation_id = ?", (simulation_id,))
            conn.execute("DELETE FROM runs WHERE simulation_id = ?", (simulation_id,))
        finally:
            conn.close()

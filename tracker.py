import sqlite3
import time
import uuid


class Tracker:
    def __init__(self, db_path="data/eval.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS problems (
                idx INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                model_label TEXT NOT NULL,
                problem_text TEXT,
                gold_answer TEXT,
                problem_type TEXT,
                level TEXT,
                PRIMARY KEY (idx, run_id)
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                problem_idx INTEGER NOT NULL,
                model_label TEXT NOT NULL,
                sample_num INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                response_text TEXT,
                extracted_answer TEXT,
                gold_answer TEXT,
                correct INTEGER,
                error_message TEXT,
                started_at REAL,
                finished_at REAL,
                UNIQUE(run_id, problem_idx, sample_num)
            );
        """)

    def new_run(self) -> str:
        return uuid.uuid4().hex[:12]

    def register_problem(self, run_id, problem):
        self.conn.execute(
            "INSERT OR IGNORE INTO problems (idx, run_id, model_label, problem_text, gold_answer, problem_type, level) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (problem["idx"], run_id, problem.get("model_label", ""), problem["problem"], problem["gold_answer"], problem.get("type", ""), problem.get("level", "")),
        )
        self.conn.commit()

    def enqueue(self, run_id, problem_idx, model_label, sample_num, gold_answer) -> int:
        cur = self.conn.execute(
            "INSERT INTO samples (run_id, problem_idx, model_label, sample_num, status, gold_answer) VALUES (?, ?, ?, ?, 'pending', ?)",
            (run_id, problem_idx, model_label, sample_num, gold_answer),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_in_progress(self, row_id):
        self.conn.execute(
            "UPDATE samples SET status='in_progress', started_at=? WHERE id=?",
            (time.time(), row_id),
        )
        self.conn.commit()

    def mark_complete(self, row_id, response_text, extracted_answer, correct, status):
        self.conn.execute(
            "UPDATE samples SET status=?, response_text=?, extracted_answer=?, correct=?, finished_at=? WHERE id=?",
            (status, response_text, extracted_answer, int(correct), time.time(), row_id),
        )
        self.conn.commit()

    def mark_error(self, row_id, error_message):
        self.conn.execute(
            "UPDATE samples SET status='error', error_message=?, finished_at=? WHERE id=?",
            (error_message, time.time(), row_id),
        )
        self.conn.commit()

    def get_problem_samples(self, run_id, problem_idx) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM samples WHERE run_id=? AND problem_idx=? ORDER BY sample_num",
            (run_id, problem_idx),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_data(self, run_id=None) -> dict:
        """Return all data grouped by model_label, shaped for dashboard consumption."""
        if run_id:
            problems = self.conn.execute("SELECT * FROM problems WHERE run_id=? ORDER BY idx", (run_id,)).fetchall()
            samples = self.conn.execute("SELECT * FROM samples WHERE run_id=? ORDER BY problem_idx, sample_num", (run_id,)).fetchall()
        else:
            problems = self.conn.execute("SELECT * FROM problems ORDER BY run_id, idx").fetchall()
            samples = self.conn.execute("SELECT * FROM samples ORDER BY run_id, problem_idx, sample_num").fetchall()

        # Group samples by (model_label, problem_idx)
        sample_map = {}
        for s in samples:
            s = dict(s)
            key = (s["model_label"], s["problem_idx"])
            sample_map.setdefault(key, []).append(s)

        # Build per-model results
        results = {}
        for p in problems:
            p = dict(p)
            label = p["model_label"]
            samps = sample_map.get((label, p["idx"]), [])
            finished = [s for s in samps if s["status"] in ("correct", "incorrect")]
            num_correct = sum(1 for s in samps if s["status"] == "correct")
            num_total = len(samps)
            accuracy = num_correct / len(finished) if finished else 0

            results.setdefault(label, []).append({
                "idx": p["idx"],
                "problem": p["problem_text"],
                "gold_answer": p["gold_answer"],
                "type": p["problem_type"],
                "level": p["level"],
                "num_correct": num_correct,
                "num_total": num_total,
                "accuracy": accuracy,
                "per_sample": [
                    {
                        "status": s["status"],
                        "extracted_answer": s["extracted_answer"],
                        "correct": bool(s["correct"]) if s["correct"] is not None else None,
                        "raw_response": s["response_text"] or "",
                        "error": s["error_message"],
                        "started_at": s["started_at"],
                        "finished_at": s["finished_at"],
                    }
                    for s in samps
                ],
            })

        return results

    def get_live_counts(self, run_id=None) -> dict:
        """Return counts by status for the dashboard."""
        where = "WHERE run_id=?" if run_id else ""
        params = (run_id,) if run_id else ()
        rows = self.conn.execute(f"SELECT status, COUNT(*) as cnt FROM samples {where} GROUP BY status", params).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

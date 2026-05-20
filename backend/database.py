import sqlite3
from datetime import datetime

DB_PATH = "crowd_stats.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crowd_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frame_id INTEGER,
                person_count REAL,
                density_score REAL,
                density_category TEXT,
                active_tracks INTEGER,
                alert TEXT
            )
        """)


def insert_stat(frame_id, person_count, density_score, density_category, active_tracks, alert=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO crowd_stats (timestamp, frame_id, person_count, density_score, "
            "density_category, active_tracks, alert) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), frame_id, person_count,
             density_score, density_category, active_tracks, alert)
        )


def get_recent_stats(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT timestamp, frame_id, person_count, density_score, density_category, "
            "active_tracks, alert FROM crowd_stats ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {"timestamp": r[0], "frame_id": r[1], "person_count": r[2],
         "density_score": r[3], "density_category": r[4],
         "active_tracks": r[5], "alert": r[6]}
        for r in rows
    ]


def get_summary():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*), AVG(person_count), MAX(person_count), MIN(person_count), "
            "SUM(CASE WHEN alert IS NOT NULL THEN 1 ELSE 0 END) FROM crowd_stats"
        ).fetchone()
        cat_rows = conn.execute(
            "SELECT density_category, COUNT(*) FROM crowd_stats GROUP BY density_category"
        ).fetchall()

    if not row or row[0] == 0:
        return {}

    return {
        "total_records": row[0],
        "avg_count": round(row[1], 2),
        "max_count": round(row[2], 2),
        "min_count": round(row[3], 2),
        "alerts_triggered": row[4],
        "category_distribution": {r[0]: r[1] for r in cat_rows},
    }

"""
Persistent storage for saved patient assessments.

A small SQLite database keeps every completed assessment on disk so records
survive application restarts and are never overwritten: re-assessing a patient
appends a new timestamped row instead of replacing the previous one.

The module is deliberately dependency-free (``sqlite3`` ships with Python) and
stores the raw inputs, both predictions, the probability distributions and the
SHAP explanations so a saved dashboard can be reconstructed without re-running
the models.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import sqlite3

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "matcare.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    age REAL,
    gestational_age TEXT,
    notes TEXT,
    maternal_inputs TEXT NOT NULL,
    cbc_inputs TEXT NOT NULL,
    maternal_prediction TEXT,
    maternal_confidence REAL,
    maternal_probabilities TEXT,
    anemia_prediction TEXT,
    anemia_confidence REAL,
    anemia_probabilities TEXT,
    shap_maternal TEXT,
    shap_anemia TEXT,
    maternal_weights TEXT,
    anemia_weights TEXT
);
CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments (patient_id);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _connect():
    """Yield a connection that is always committed and closed.

    ``with sqlite3.connect(...)`` commits on exit but leaves the file handle
    open, which locks the database on Windows. Closing explicitly keeps the
    database file releasable and safe to back up or move.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    """Create the database file and table if they do not exist yet."""
    with _connect() as connection:
        connection.executescript(SCHEMA)


def _fallback(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _dumps(value):
    return json.dumps(value, default=_fallback)


def _loads(value):
    return json.loads(value) if value else None


def _row_to_record(row):
    record = dict(row)
    for key in ("maternal_inputs", "cbc_inputs", "maternal_probabilities",
                "anemia_probabilities", "shap_maternal", "shap_anemia",
                "maternal_weights", "anemia_weights"):
        if key in record:
            record[key] = _loads(record[key])
    return record


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def save_assessment(patient_id, patient_name, maternal_inputs, cbc_inputs,
                    maternal_result, anemia_result, shap=None, age=None,
                    gestational_age=None, notes="", created_at=None):
    """Insert one assessment and return its new row id.

    ``maternal_result`` and ``anemia_result`` are the dictionaries returned by
    ``app.inference.predict_module``. ``shap`` optionally maps ``maternal`` and
    ``anemia`` to SHAP DataFrames, which are stored as plain records.
    """
    shap = shap or {}
    timestamp = created_at or datetime.datetime.now().isoformat(timespec="seconds")

    def _frame_records(frame):
        if frame is None:
            return None
        if hasattr(frame, "to_dict"):
            return frame.to_dict("records")
        return frame

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO assessments (
                patient_id, patient_name, created_at, age, gestational_age, notes,
                maternal_inputs, cbc_inputs,
                maternal_prediction, maternal_confidence, maternal_probabilities,
                anemia_prediction, anemia_confidence, anemia_probabilities,
                shap_maternal, shap_anemia, maternal_weights, anemia_weights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(patient_id), str(patient_name), timestamp, age, gestational_age, notes,
                _dumps(maternal_inputs), _dumps(cbc_inputs),
                maternal_result.get("predicted_class"),
                float(maternal_result.get("confidence", 0.0)),
                _dumps({k: list(map(float, v)) for k, v in maternal_result["probabilities"].items()}),
                anemia_result.get("predicted_class"),
                float(anemia_result.get("confidence", 0.0)),
                _dumps({k: list(map(float, v)) for k, v in anemia_result["probabilities"].items()}),
                _dumps(_frame_records(shap.get("maternal"))),
                _dumps(_frame_records(shap.get("anemia"))),
                _dumps({"w_rf": float(maternal_result.get("w_rf", 0.5)),
                        "w_xgb": float(maternal_result.get("w_xgb", 0.5))}),
                _dumps({"w_rf": float(anemia_result.get("w_rf", 0.5)),
                        "w_xgb": float(anemia_result.get("w_xgb", 0.5))}),
            ),
        )
        return int(cursor.lastrowid)


def _norm_inputs(container):
    """Normalise an input mapping so ints and floats compare equal."""
    result = {}
    for key, value in (container or {}).items():
        try:
            result[key] = round(float(value), 6)
        except (TypeError, ValueError):
            result[key] = value
    return result


def find_duplicate(patient_id, maternal_inputs, cbc_inputs, within_seconds=30):
    """Return the id of a just-saved identical assessment, or ``None``.

    Only the patient's most recent row is considered, and only when it was
    written within ``within_seconds`` with byte-identical inputs. This swallows
    an accidental repeated submission without merging genuine repeat assessments
    that a clinician records later.
    """
    with _connect() as connection:
        row = connection.execute(
            "SELECT id, created_at, maternal_inputs, cbc_inputs FROM assessments "
            "WHERE patient_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (str(patient_id),),
        ).fetchone()
    if row is None:
        return None
    try:
        created = datetime.datetime.fromisoformat(row["created_at"])
    except (TypeError, ValueError):
        return None
    if (datetime.datetime.now() - created).total_seconds() > within_seconds:
        return None
    if _norm_inputs(_loads(row["maternal_inputs"])) != _norm_inputs(maternal_inputs):
        return None
    if _norm_inputs(_loads(row["cbc_inputs"])) != _norm_inputs(cbc_inputs):
        return None
    return int(row["id"])


def update_record(record_id, patient_name=None, notes=None):
    """Update the editable metadata of a saved record (name / notes)."""
    fields, values = [], []
    if patient_name is not None:
        fields.append("patient_name = ?")
        values.append(str(patient_name))
    if notes is not None:
        fields.append("notes = ?")
        values.append(str(notes))
    if not fields:
        return
    values.append(int(record_id))
    with _connect() as connection:
        connection.execute(f"UPDATE assessments SET {', '.join(fields)} WHERE id = ?", values)


def delete_record(record_id):
    """Delete one assessment by id."""
    with _connect() as connection:
        connection.execute("DELETE FROM assessments WHERE id = ?", (int(record_id),))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def get_record(record_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM assessments WHERE id = ?", (int(record_id),)
        ).fetchone()
    return _row_to_record(row) if row else None


def list_records(query=None, limit=None):
    """Return saved assessments newest first, optionally filtered by a search term."""
    sql = "SELECT * FROM assessments"
    params = []
    if query:
        sql += " WHERE patient_name LIKE ? OR patient_id LIKE ?"
        term = f"%{query}%"
        params.extend([term, term])
    sql += " ORDER BY created_at DESC, id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_row_to_record(row) for row in rows]


def get_history(patient_id):
    """Return every assessment for one patient, oldest first (for trend charts)."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM assessments WHERE patient_id = ? ORDER BY created_at ASC, id ASC",
            (str(patient_id),),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def get_latest_record(patient_id):
    """Return the most recent assessment for one patient, or ``None``."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM assessments WHERE patient_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (str(patient_id),),
        ).fetchone()
    return _row_to_record(row) if row else None


def list_patients(query=None):
    """Return one summary per patient, most recently assessed first.

    Each entry has ``patient_id``, ``assessment_count``, ``last_assessment`` and
    the decoded ``latest`` record so a profile card can be drawn without a
    second query. ``query`` filters on patient name or ID.
    """
    with _connect() as connection:
        if query:
            term = f"%{query}%"
            rows = connection.execute(
                "SELECT DISTINCT patient_id FROM assessments "
                "WHERE patient_name LIKE ? OR patient_id LIKE ?",
                (term, term),
            ).fetchall()
        else:
            rows = connection.execute("SELECT DISTINCT patient_id FROM assessments").fetchall()

        patients = []
        for row in rows:
            patient_id = row["patient_id"]
            history = connection.execute(
                "SELECT * FROM assessments WHERE patient_id = ? "
                "ORDER BY created_at DESC, id DESC",
                (patient_id,),
            ).fetchall()
            latest = _row_to_record(history[0])
            patients.append({
                "patient_id": patient_id,
                "assessment_count": len(history),
                "last_assessment": latest["created_at"],
                "latest": latest,
            })

    patients.sort(key=lambda item: (item["last_assessment"], item["latest"]["id"]), reverse=True)
    return patients


def count_records():
    with _connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM assessments").fetchone()[0])


def count_patients():
    """Number of distinct patient IDs with at least one saved assessment."""
    with _connect() as connection:
        return int(
            connection.execute("SELECT COUNT(DISTINCT patient_id) FROM assessments").fetchone()[0]
        )

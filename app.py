"""
app.py - Shop Floor Scheduler (a lightweight, JobBOSS²-inspired scheduling board)
==================================================================================

A single-file Streamlit application with three connected views:

    1. Scheduler Whiteboard  - load vs. capacity, Gantt chart, and a
                               "Scheduling Advisor" that raises RED flags.
    2. Shop Floor            - tablet/phone friendly task list for an operator
                               or a machine, with one-tap status updates.
    3. Admin / Re-Prioritize - override job priority/due date, reassign
                               operations to other machines or operators,
                               edit machine capacity/status, add new jobs.

Data layer
----------
SQLite with three related tables (machines -> tasks <- jobs). By default the
database lives in memory and is re-seeded with demo data every time the server
starts. Set the environment variable SHOP_DB_PATH to a file path (for example
``shop_schedule.db``) to make the data persistent between restarts.

The connection is created once per server process (``st.cache_resource``), so
every browser/tablet connected to the app sees the same live schedule.

Scheduling logic
----------------
Forward, finite-capacity, priority-based dispatching:

    * Operations already "In Progress" are loaded first (they occupy the machine).
    * Remaining operations are loaded in order of job Priority (1 = highest),
      then Due Date, then Job ID, then operation Sequence.
    * An operation can't start until (a) its machine has an open slot long
      enough for it and (b) the job's previous operation has finished.
      Lower-priority work may backfill idle gaps left between higher-priority
      operations, but never delays them.
    * Duration (in working days) = remaining hours / machine daily capacity.
    * Working days are Monday-Friday.
    * Machines with status "Down" or "Maintenance" can't be scheduled; any job
      routed through them is flagged.

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration / constants
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Shop Floor Scheduler", layout="wide")

DB_PATH = os.environ.get("SHOP_DB_PATH", ":memory:")

TASK_STATUSES = ["Not Started", "In Progress", "Complete"]
MACHINE_STATUSES = ["Running", "Idle", "Down", "Maintenance"]
UNAVAILABLE_MACHINE_STATUSES = {"Down", "Maintenance"}
OPERATORS = ["Mike T.", "Dana R.", "Chris P.", "Jen K.", "Luis M.", "Pat S."]

# Gantt display only: one working day is drawn from 07:00 to 23:00 (2 shifts).
SHIFT_START = time(7, 0)
DISPLAY_DAY_HOURS = 16

RED = "#C62828"
GREEN = "#2E7D32"
GREY = "#90A4AE"

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id            TEXT PRIMARY KEY,
    machine_name          TEXT NOT NULL,
    daily_capacity_hours  REAL NOT NULL CHECK (daily_capacity_hours >= 0 AND daily_capacity_hours <= 24),
    current_status        TEXT NOT NULL DEFAULT 'Running'
                          CHECK (current_status IN ('Running','Idle','Down','Maintenance'))
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id                 TEXT PRIMARY KEY,
    customer               TEXT NOT NULL,
    part_name              TEXT NOT NULL,
    quantity               INTEGER NOT NULL CHECK (quantity > 0),
    total_estimated_hours  REAL NOT NULL DEFAULT 0,      -- rolled up from tasks
    current_operation      TEXT,                         -- rolled up from tasks
    status                 TEXT NOT NULL DEFAULT 'Released',
    due_date               TEXT NOT NULL,                -- ISO yyyy-mm-dd
    priority               INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id             TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    machine_id         TEXT NOT NULL REFERENCES machines(machine_id),
    sequence           INTEGER NOT NULL,
    required_hours     REAL NOT NULL CHECK (required_hours >= 0),
    status             TEXT NOT NULL DEFAULT 'Not Started'
                       CHECK (status IN ('Not Started','In Progress','Complete')),
    assigned_operator  TEXT,
    UNIQUE (job_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_tasks_machine  ON tasks(machine_id);
CREATE INDEX IF NOT EXISTS idx_tasks_operator ON tasks(assigned_operator);
"""


# --------------------------------------------------------------------------- #
# Working-day helpers
# --------------------------------------------------------------------------- #
def add_workdays(start: date, n: int) -> date:
    """Return the date `n` working days (Mon-Fri) from `start` (n may be negative)."""
    rolled = np.busday_offset(np.datetime64(start, "D"), 0, roll="forward")
    return pd.Timestamp(np.busday_offset(rolled, n)).date()


def offset_to_datetime(start: date, offset: float, is_finish: bool = False) -> datetime:
    """Convert a fractional working-day offset (0 = start of `start`) to a datetime.

    A finish that lands exactly on a day boundary is shown at the END of the
    previous working day rather than the start of the next one.
    """
    whole = math.floor(offset)
    frac = offset - whole
    if is_finish and frac == 0 and whole > 0:
        whole, frac = whole - 1, 1.0
    day = add_workdays(start, whole)
    return datetime.combine(day, SHIFT_START) + timedelta(hours=frac * DISPLAY_DAY_HOURS)


def ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# --------------------------------------------------------------------------- #
# Data access layer
# --------------------------------------------------------------------------- #
class ShopDB:
    """Thin, thread-safe wrapper around a SQLite connection.

    All writes go through this class so job roll-up fields (Total Estimated
    Hours, Current Operation, Status) always stay in sync with the tasks table.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.RLock()
        with self.lock, self.conn:
            self.conn.executescript(SCHEMA)
        if self._count("jobs") == 0:
            self.seed_demo_data()

    # ---- generic helpers -------------------------------------------------- #
    def _count(self, table: str) -> int:
        with self.lock:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.lock:
            return pd.read_sql_query(sql, self.conn, params=params)

    # ---- reads ------------------------------------------------------------ #
    def machines(self) -> pd.DataFrame:
        return self.query("SELECT * FROM machines ORDER BY machine_id")

    def jobs(self) -> pd.DataFrame:
        df = self.query("SELECT * FROM jobs ORDER BY priority, due_date, job_id")
        df["due_date"] = pd.to_datetime(df["due_date"]).dt.date
        return df

    def tasks(self) -> pd.DataFrame:
        return self.query(
            """SELECT t.*, m.machine_name
                 FROM tasks t JOIN machines m ON m.machine_id = t.machine_id
             ORDER BY t.job_id, t.sequence"""
        )

    # ---- writes ----------------------------------------------------------- #
    def update_task_status(self, task_id: int, status: str) -> None:
        if status not in TASK_STATUSES:
            raise ValueError(f"Invalid task status: {status}")
        with self.lock, self.conn:
            self.conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (status, int(task_id)))
            job_id = self.conn.execute(
                "SELECT job_id FROM tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()[0]
            self._sync_job(job_id)

    def update_task(self, task_id: int, machine_id: str, operator: str | None,
                    required_hours: float, status: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """UPDATE tasks SET machine_id = ?, assigned_operator = ?,
                                    required_hours = ?, status = ?
                    WHERE task_id = ?""",
                (machine_id, operator or None, float(required_hours), status, int(task_id)),
            )
            job_id = self.conn.execute(
                "SELECT job_id FROM tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()[0]
            self._sync_job(job_id)

    def move_job_operations(self, job_id: str, from_machine: str, to_machine: str) -> int:
        """Move every open (not Complete) operation of a job from one machine to another."""
        with self.lock, self.conn:
            cur = self.conn.execute(
                """UPDATE tasks SET machine_id = ?
                    WHERE job_id = ? AND machine_id = ? AND status <> 'Complete'""",
                (to_machine, job_id, from_machine),
            )
            self._sync_job(job_id)
            return cur.rowcount

    def update_job(self, job_id: str, priority: int, due_date: date) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE jobs SET priority = ?, due_date = ? WHERE job_id = ?",
                (int(priority), due_date.isoformat(), job_id),
            )

    def update_machine(self, machine_id: str, capacity: float, status: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE machines SET daily_capacity_hours = ?, current_status = ? WHERE machine_id = ?",
                (float(capacity), status, machine_id),
            )

    def add_job(self, job: dict, routing: list[dict]) -> None:
        """Insert a job plus its routing (list of {machine_id, required_hours, assigned_operator})."""
        with self.lock, self.conn:
            self.conn.execute(
                """INSERT INTO jobs (job_id, customer, part_name, quantity, due_date, priority)
                   VALUES (:job_id, :customer, :part_name, :quantity, :due_date, :priority)""",
                job,
            )
            for seq, op in enumerate(routing, start=1):
                self.conn.execute(
                    """INSERT INTO tasks (job_id, machine_id, sequence, required_hours, assigned_operator)
                       VALUES (?, ?, ?, ?, ?)""",
                    (job["job_id"], op["machine_id"], seq * 10, float(op["required_hours"]),
                     op.get("assigned_operator") or None),
                )
            self._sync_job(job["job_id"])

    def reset(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript("DELETE FROM tasks; DELETE FROM jobs; DELETE FROM machines;")
        self.seed_demo_data()

    # ---- roll-ups --------------------------------------------------------- #
    def _sync_job(self, job_id: str) -> None:
        """Recalculate the job's roll-up fields from its tasks. Caller holds the lock."""
        rows = self.conn.execute(
            """SELECT t.sequence, t.status, t.required_hours, m.machine_name
                 FROM tasks t JOIN machines m ON m.machine_id = t.machine_id
                WHERE t.job_id = ? ORDER BY t.sequence""",
            (job_id,),
        ).fetchall()
        total_hours = sum(r[2] for r in rows)
        open_ops = [r for r in rows if r[1] != "Complete"]
        if not rows:
            status, current_op = "Open", "No routing"
        elif not open_ops:
            status, current_op = "Complete", "All operations complete"
        else:
            nxt = open_ops[0]
            current_op = f"Op {nxt[0]} - {nxt[3]}"
            status = "In Progress" if any(r[1] != "Not Started" for r in rows) else "Released"
        self.conn.execute(
            """UPDATE jobs SET total_estimated_hours = ?, current_operation = ?, status = ?
                WHERE job_id = ?""",
            (total_hours, current_op, status, job_id),
        )

    def sync_all_jobs(self) -> None:
        with self.lock, self.conn:
            for (job_id,) in self.conn.execute("SELECT job_id FROM jobs").fetchall():
                self._sync_job(job_id)

    # ---- demo data -------------------------------------------------------- #
    def seed_demo_data(self) -> None:
        """Powder-metal shop demo data. Due dates are relative to today."""
        today = date.today()
        machines = [
            ("PRS-01", "Compaction Press 1 (150 T)", 16, "Running"),
            ("PRS-02", "Compaction Press 2 (300 T)", 16, "Running"),
            ("FUR-01", "Mesh Belt Sintering Furnace", 24, "Running"),
            ("SIZ-01", "Sizing / Coining Press", 8, "Running"),
            ("CNC-01", "CNC Lathe", 8, "Running"),
            ("INS-01", "Final Inspection", 8, "Idle"),
        ]
        # (job_id, customer, part, qty, due offset in workdays, priority,
        #   [(machine, hours, status, operator), ...])
        jobs = [
            ("J-1001", "Acme Automotive", "Bushing FC-0208", 5000, 4, 2,
             [("PRS-01", 6, "Not Started", "Mike T."), ("FUR-01", 8, "Not Started", "Dana R."),
              ("SIZ-01", 10, "Not Started", "Chris P."), ("INS-01", 2, "Not Started", "Pat S.")]),
            ("J-1002", "Midwest Hydraulics", "Gear FN-0205", 2500, 4, 1,
             [("PRS-02", 10, "In Progress", "Jen K."), ("FUR-01", 10, "Not Started", "Dana R."),
              ("SIZ-01", 10, "Not Started", "Chris P."), ("CNC-01", 8, "Not Started", "Luis M."),
              ("INS-01", 3, "Not Started", "Pat S.")]),
            ("J-1003", "Northern Tool Co.", "Cam Lobe FL-4405", 1200, 6, 3,
             [("PRS-02", 8, "Not Started", "Jen K."), ("FUR-01", 6, "Not Started", "Dana R."),
              ("CNC-01", 10, "Not Started", "Luis M."), ("INS-01", 2, "Not Started", "Pat S.")]),
            ("J-1004", "Allegheny Lock", "Latch Plate F-0005", 10000, 1, 2,
             [("PRS-01", 12, "Not Started", "Mike T."), ("FUR-01", 12, "Not Started", "Dana R."),
              ("SIZ-01", 16, "Not Started", "Chris P."), ("INS-01", 3, "Not Started", "Pat S.")]),
            ("J-1005", "Keystone Appliance", "Sprocket FC-0205", 4000, 8, 4,
             [("PRS-01", 9, "Not Started", "Mike T."), ("FUR-01", 9, "Not Started", "Dana R."),
              ("SIZ-01", 6, "Not Started", "Chris P."), ("INS-01", 2, "Not Started", "Pat S.")]),
            ("J-1006", "River Valley Marine", "316L Bracket", 800, 7, 3,
             [("PRS-02", 6, "Not Started", "Jen K."), ("FUR-01", 10, "Not Started", "Dana R."),
              ("SIZ-01", 6, "Not Started", "Chris P."), ("CNC-01", 6, "Not Started", "Luis M."),
              ("INS-01", 2, "Not Started", "Pat S.")]),
            ("J-1007", "Acme Automotive", "Oil Pump Rotor", 6000, 10, 5,
             [("PRS-02", 14, "Not Started", "Jen K."), ("FUR-01", 12, "Not Started", "Dana R."),
              ("SIZ-01", 8, "Not Started", "Chris P."), ("CNC-01", 12, "Not Started", "Luis M."),
              ("INS-01", 4, "Not Started", "Pat S.")]),
            ("J-1008", "Summit Ag Equipment", "Pulley Hub FN-0208", 1500, -1, 1,
             [("PRS-01", 5, "Complete", "Mike T."), ("FUR-01", 6, "Complete", "Dana R."),
              ("SIZ-01", 6, "In Progress", "Chris P."), ("INS-01", 2, "Not Started", "Pat S.")]),
        ]
        with self.lock, self.conn:
            self.conn.executemany("INSERT INTO machines VALUES (?, ?, ?, ?)", machines)
            for job_id, cust, part, qty, due_off, pri, routing in jobs:
                self.conn.execute(
                    """INSERT INTO jobs (job_id, customer, part_name, quantity, due_date, priority)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (job_id, cust, part, qty, add_workdays(today, due_off).isoformat(), pri),
                )
                for seq, (mach, hrs, status, op) in enumerate(routing, start=1):
                    self.conn.execute(
                        """INSERT INTO tasks (job_id, machine_id, sequence, required_hours,
                                              status, assigned_operator)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (job_id, mach, seq * 10, hrs, status, op),
                    )
        self.sync_all_jobs()


@st.cache_resource
def get_db() -> ShopDB:
    """One shared database per server process (all users see the same schedule)."""
    return ShopDB(DB_PATH)


# --------------------------------------------------------------------------- #
# Scheduling engine
# --------------------------------------------------------------------------- #
def earliest_slot(intervals: list[tuple[float, float]], ready: float, duration: float) -> float:
    """Earliest start >= `ready` where `duration` fits between already-booked intervals.

    This lets lower-priority work backfill idle gaps without delaying anything
    that was booked earlier (higher priority).
    """
    candidate = ready
    for b_start, b_end in intervals:          # intervals are sorted by start
        if candidate + duration <= b_start + 1e-9:
            break                             # fits in the gap before this booking
        candidate = max(candidate, b_end)
    return candidate


def build_schedule(jobs: pd.DataFrame, tasks: pd.DataFrame, machines: pd.DataFrame,
                   start: date) -> pd.DataFrame:
    """Forward finite-capacity schedule. Returns tasks with start/finish offsets and datetimes."""
    cap = machines.set_index("machine_id")["daily_capacity_hours"].to_dict()
    m_status = machines.set_index("machine_id")["current_status"].to_dict()

    t = tasks.merge(jobs[["job_id", "priority", "due_date", "part_name", "customer"]], on="job_id")
    t["remaining_hours"] = np.where(t["status"] == "Complete", 0.0, t["required_hours"])
    t["run_first"] = np.where(t["status"] == "In Progress", 0, 1)
    t = t.sort_values(["run_first", "priority", "due_date", "job_id", "sequence"])

    busy: dict[str, list[tuple[float, float]]] = {m: [] for m in cap}  # booked intervals per machine
    job_ready: dict[str, float] = {}       # when the job's previous op finishes
    results = []

    for r in t.itertuples(index=False):
        ready = job_ready.get(r.job_id, 0.0)
        if r.status == "Complete":
            results.append((r.task_id, np.nan, np.nan, False))
            continue
        daily_cap = cap.get(r.machine_id, 0.0)
        if daily_cap <= 0 or m_status.get(r.machine_id) in UNAVAILABLE_MACHINE_STATUSES or math.isinf(ready):
            results.append((r.task_id, np.nan, np.nan, True))
            job_ready[r.job_id] = math.inf     # everything downstream is blocked too
            continue
        duration = r.remaining_hours / daily_cap
        s = earliest_slot(busy[r.machine_id], ready, duration)
        f = s + duration
        busy[r.machine_id].append((s, f))
        busy[r.machine_id].sort()
        job_ready[r.job_id] = f
        results.append((r.task_id, s, f, False))

    res = pd.DataFrame(results, columns=["task_id", "start_offset", "finish_offset", "blocked"])
    t = t.merge(res, on="task_id")
    t["start_dt"] = [offset_to_datetime(start, s) if pd.notna(s) else pd.NaT for s in t["start_offset"]]
    t["finish_dt"] = [offset_to_datetime(start, f, True) if pd.notna(f) else pd.NaT for f in t["finish_offset"]]
    return t.sort_values(["machine_id", "start_offset"]).reset_index(drop=True)


def job_completion_report(jobs: pd.DataFrame, sched: pd.DataFrame, today: date) -> pd.DataFrame:
    """Estimated completion and late/blocked flags per job."""
    rows = []
    for j in jobs.itertuples(index=False):
        jt = sched[sched["job_id"] == j.job_id]
        open_t = jt[jt["status"] != "Complete"]
        if open_t.empty:
            est, blocked = None, False
        elif open_t["blocked"].any():
            est, blocked = None, True
        else:
            est, blocked = pd.Timestamp(open_t["finish_dt"].max()).date(), False

        if j.status == "Complete":
            flag, days_late = "Complete", 0
        elif blocked:
            flag, days_late = "BLOCKED - machine unavailable", None
        elif est is not None and est > j.due_date:
            flag, days_late = "LATE", int(np.busday_count(j.due_date, est))
        elif j.due_date < today:
            flag, days_late = "LATE", int(np.busday_count(j.due_date, today))
        else:
            flag, days_late = "On Time", 0
        rows.append({
            "job_id": j.job_id, "est_completion": est, "days_late": days_late,
            "advisor_flag": flag, "remaining_hours": float(open_t["remaining_hours"].sum()),
        })
    return jobs.merge(pd.DataFrame(rows), on="job_id")


def capacity_report(machines: pd.DataFrame, sched: pd.DataFrame, jobs: pd.DataFrame,
                    today: date, horizon_days: int) -> pd.DataFrame:
    """Loaded hours (open work due within the horizon, incl. overdue) vs. available hours."""
    horizon_end = add_workdays(today, horizon_days - 1)
    open_t = sched[sched["status"] != "Complete"]
    in_horizon = open_t[open_t["due_date"] <= horizon_end]
    loaded = in_horizon.groupby("machine_id")["remaining_hours"].sum()
    backlog = open_t.groupby("machine_id")["remaining_hours"].sum()

    rep = machines.copy()
    rep["available_hours"] = np.where(
        rep["current_status"].isin(UNAVAILABLE_MACHINE_STATUSES), 0.0,
        rep["daily_capacity_hours"] * horizon_days,
    )
    rep["loaded_hours"] = rep["machine_id"].map(loaded).fillna(0.0)
    rep["total_backlog_hours"] = rep["machine_id"].map(backlog).fillna(0.0)
    rep["utilization_pct"] = np.where(
        rep["available_hours"] > 0, rep["loaded_hours"] / rep["available_hours"] * 100,
        np.where(rep["loaded_hours"] > 0, np.inf, 0.0),
    )
    rep["over_capacity"] = rep["loaded_hours"] > rep["available_hours"]
    rep["overload_hours"] = (rep["loaded_hours"] - rep["available_hours"]).clip(lower=0)
    return rep


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
.red-flag {{background:{RED};color:#fff;padding:.75rem 1rem;border-radius:.5rem;
            margin-bottom:.5rem;font-weight:600;font-size:1.02rem;}}
.green-ok {{background:{GREEN};color:#fff;padding:.75rem 1rem;border-radius:.5rem;font-weight:600;}}
.task-card-title {{font-size:1.35rem;font-weight:700;margin-bottom:.15rem;}}
.task-rank {{display:inline-block;background:#1565C0;color:#fff;border-radius:.4rem;
             padding:.1rem .6rem;margin-right:.5rem;}}
div.stButton > button {{min-height:3rem;font-size:1.05rem;font-weight:600;}}
</style>
"""


def red_flag(text: str) -> None:
    st.markdown(f'<div class="red-flag">RED FLAG &nbsp;|&nbsp; {text}</div>', unsafe_allow_html=True)


def style_job_rows(row: pd.Series) -> list[str]:
    if row["Advisor"] in ("LATE",) or str(row["Advisor"]).startswith("BLOCKED"):
        return [f"background-color:{RED};color:white;font-weight:600"] * len(row)
    if row["Advisor"] == "Complete":
        return ["color:#607D8B"] * len(row)
    return [""] * len(row)


def style_machine_rows(row: pd.Series) -> list[str]:
    if row["Over Capacity"]:
        return [f"background-color:{RED};color:white;font-weight:600"] * len(row)
    return [""] * len(row)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def view_whiteboard(machines, jobs_rep, cap_rep, sched, horizon_days) -> None:
    st.title("Scheduler Whiteboard")
    st.caption(f"Planning horizon: next {horizon_days} working days. Priority 1 = highest.")

    open_jobs = jobs_rep[jobs_rep["status"] != "Complete"]
    late = jobs_rep[jobs_rep["advisor_flag"] == "LATE"]
    blocked = jobs_rep[jobs_rep["advisor_flag"].str.startswith("BLOCKED")]
    over = cap_rep[cap_rep["over_capacity"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Jobs", len(open_jobs))
    c2.metric("Late / Blocked Jobs", len(late) + len(blocked))
    c3.metric("Over-Capacity Machines", len(over))
    c4.metric("Open Hours (all work)", f"{cap_rep['total_backlog_hours'].sum():,.1f}")

    # ---- Scheduling Advisor --------------------------------------------- #
    st.subheader("Scheduling Advisor")
    if late.empty and blocked.empty and over.empty:
        st.markdown('<div class="green-ok">All jobs on time and all machines within capacity.</div>',
                    unsafe_allow_html=True)
    for j in late.itertuples(index=False):
        est = j.est_completion.strftime("%a %b %d") if j.est_completion else "not yet scheduled"
        red_flag(f"Job {j.job_id} ({j.customer} - {j.part_name}) is LATE: due "
                 f"{j.due_date:%a %b %d}, estimated completion {est} "
                 f"({j.days_late} working day(s) late). Priority {j.priority}.")
    for j in blocked.itertuples(index=False):
        red_flag(f"Job {j.job_id} ({j.part_name}) can't be scheduled: it is routed through a "
                 f"machine that is Down / in Maintenance. Reassign it on the Admin page.")
    for m in over.itertuples(index=False):
        util = "no capacity available" if math.isinf(m.utilization_pct) else f"{m.utilization_pct:.0f}% loaded"
        red_flag(f"Machine {m.machine_id} ({m.machine_name}) is OVER CAPACITY: "
                 f"{m.loaded_hours:.1f} h loaded vs {m.available_hours:.1f} h available "
                 f"({util}, {m.overload_hours:.1f} h over).")

    # ---- Load vs capacity chart ----------------------------------------- #
    st.subheader("Loaded Hours vs. Available Capacity")
    fig = go.Figure()
    fig.add_bar(name="Available Hours", x=cap_rep["machine_id"], y=cap_rep["available_hours"],
                marker_color=GREY)
    fig.add_bar(name="Loaded Hours", x=cap_rep["machine_id"], y=cap_rep["loaded_hours"],
                marker_color=[RED if o else "#1565C0" for o in cap_rep["over_capacity"]],
                text=[f"{v:.0f} h" for v in cap_rep["loaded_hours"]], textposition="outside")
    fig.update_layout(barmode="group", height=380, margin=dict(t=30, b=10),
                      yaxis_title="Hours", legend=dict(orientation="h", y=1.12),
                      hovermode="x unified")
    fig.update_xaxes(ticktext=[f"{i}<br>{n}" for i, n in zip(cap_rep["machine_id"], cap_rep["machine_name"])],
                     tickvals=cap_rep["machine_id"])
    st.plotly_chart(fig, width="stretch")

    mt = cap_rep[["machine_id", "machine_name", "current_status", "daily_capacity_hours",
                  "available_hours", "loaded_hours", "utilization_pct", "total_backlog_hours",
                  "over_capacity"]].rename(columns={
        "machine_id": "Machine ID", "machine_name": "Machine Name", "current_status": "Status",
        "daily_capacity_hours": "Daily Cap (h)", "available_hours": "Available (h)",
        "loaded_hours": "Loaded (h)", "utilization_pct": "Utilization %",
        "total_backlog_hours": "Total Backlog (h)", "over_capacity": "Over Capacity"})
    st.dataframe(
        mt.style.apply(style_machine_rows, axis=1).format(
            {"Daily Cap (h)": "{:.1f}", "Available (h)": "{:.1f}", "Loaded (h)": "{:.1f}",
             "Utilization %": lambda v: "n/a" if math.isinf(v) else f"{v:.0f}%",
             "Total Backlog (h)": "{:.1f}"}),
        width="stretch", hide_index=True)

    # ---- Job schedule ---------------------------------------------------- #
    st.subheader("Job Schedule")
    jt = jobs_rep[["priority", "job_id", "customer", "part_name", "quantity", "total_estimated_hours",
                   "remaining_hours", "current_operation", "status", "due_date", "est_completion",
                   "days_late", "advisor_flag"]].rename(columns={
        "priority": "Priority", "job_id": "Job ID", "customer": "Customer", "part_name": "Part Name",
        "quantity": "Qty", "total_estimated_hours": "Total Est. Hrs", "remaining_hours": "Remaining Hrs",
        "current_operation": "Current Operation", "status": "Status", "due_date": "Due Date",
        "est_completion": "Est. Completion", "days_late": "Days Late", "advisor_flag": "Advisor"})
    st.dataframe(
        jt.style.apply(style_job_rows, axis=1).format(
            {"Total Est. Hrs": "{:.1f}", "Remaining Hrs": "{:.1f}", "Qty": "{:,}",
             "Days Late": lambda v: "" if v is None or pd.isna(v) else f"{int(v)}",
             "Est. Completion": lambda v: "" if v is None or pd.isna(v) else f"{v:%Y-%m-%d}"}),
        width="stretch", hide_index=True)

    # ---- Gantt chart ------------------------------------------------------ #
    st.subheader("Machine Gantt (finite capacity)")
    g = sched.dropna(subset=["start_offset"]).copy()
    if g.empty:
        st.info("No open, schedulable operations.")
        return
    late_ids = set(late["job_id"])
    g["Label"] = g["job_id"] + np.where(g["job_id"].isin(late_ids), " (LATE)", "")
    g["Machine"] = g["machine_id"] + " - " + g["machine_name"]
    g["Op"] = "Op " + g["sequence"].astype(str)
    gantt = px.timeline(
        g, x_start="start_dt", x_end="finish_dt", y="Machine", color="Label",
        hover_data={"job_id": True, "Op": True, "part_name": True, "remaining_hours": ":.1f",
                    "status": True, "assigned_operator": True, "due_date": True,
                    "Label": False, "Machine": False},
        text="Op",
        category_orders={"Machine": sorted(g["Machine"].unique())},
        color_discrete_sequence=px.colors.qualitative.Set2,  # no reds: red is reserved for flags
    )
    gantt.update_yaxes(title=None)
    gantt.update_traces(marker_line_width=1, marker_line_color="white",
                        textposition="inside", insidetextanchor="middle", textangle=0)
    for tr in gantt.data:  # outline late jobs in red
        if "(LATE)" in tr.name:
            tr.marker.line.color, tr.marker.line.width = RED, 3
    gantt.update_layout(height=420, margin=dict(t=20, b=10), legend_title="Job",
                        uniformtext_minsize=9, uniformtext_mode="hide")
    st.plotly_chart(gantt, width="stretch")
    st.caption("Each working day is drawn as 07:00-23:00; weekends are skipped. "
               "Red outline = job projected late.")


def view_shop_floor(db: ShopDB, sched: pd.DataFrame, machines: pd.DataFrame) -> None:
    st.title("Shop Floor")
    mode = st.radio("Find my work by", ["Machine", "Operator"], horizontal=True)
    if mode == "Machine":
        labels = {f"{r.machine_id} - {r.machine_name}": r.machine_id for r in machines.itertuples()}
        choice = st.selectbox("Select your machine", list(labels))
        mine = sched[sched["machine_id"] == labels[choice]]
        m = machines[machines["machine_id"] == labels[choice]].iloc[0]
        if m["current_status"] in UNAVAILABLE_MACHINE_STATUSES:
            red_flag(f"{m['machine_id']} is marked {m['current_status']}. Tasks can't be scheduled here.")
    else:
        ops = sorted(set(OPERATORS) | set(sched["assigned_operator"].dropna()))
        choice = st.selectbox("Select your name", ops)
        mine = sched[sched["assigned_operator"] == choice]

    open_tasks = mine[mine["status"] != "Complete"].copy()
    # In Progress first, then schedule order (blocked tasks last), then priority.
    open_tasks["ip"] = np.where(open_tasks["status"] == "In Progress", 0, 1)
    open_tasks = open_tasks.sort_values(["ip", "start_offset", "priority", "due_date"], na_position="last")

    st.subheader(f"Work queue ({len(open_tasks)} open)")
    if open_tasks.empty:
        st.success("Nothing in your queue. Check with your supervisor.")

    for rank, r in enumerate(open_tasks.itertuples(index=False), start=1):
        prev = sched[(sched["job_id"] == r.job_id) & (sched["sequence"] < r.sequence)
                     & (sched["status"] != "Complete")]
        with st.container(border=True):
            st.markdown(
                f'<div class="task-card-title"><span class="task-rank">{ordinal(rank)}</span>'
                f'{r.job_id} &middot; {r.part_name} &middot; Op {r.sequence}</div>',
                unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**Machine:** {r.machine_id}")
            c2.write(f"**Hours:** {r.required_hours:.1f}")
            c3.write(f"**Due:** {r.due_date:%a %b %d}")
            c4.write(f"**Status:** {r.status}")
            st.caption(f"Customer: {r.customer} | Priority {r.priority} | Operator: {r.assigned_operator or 'Unassigned'}"
                       + (f" | Planned start: {pd.Timestamp(r.start_dt):%a %b %d %H:%M}" if pd.notna(r.start_dt) else ""))
            if not prev.empty:
                p = prev.sort_values("sequence").iloc[-1]
                st.warning(f"Waiting on Op {p['sequence']} at {p['machine_id']} ({p['status']}).")

            b1, b2, b3 = st.columns([1, 1, 1.4])
            if b1.button("Start", key=f"start_{r.task_id}", type="primary",
                         disabled=r.status == "In Progress", width="stretch"):
                db.update_task_status(r.task_id, "In Progress")
                st.toast(f"{r.job_id} Op {r.sequence} started")
                st.rerun()
            if b2.button("Complete", key=f"done_{r.task_id}", width="stretch"):
                db.update_task_status(r.task_id, "Complete")
                st.toast(f"{r.job_id} Op {r.sequence} complete")
                st.rerun()
            new_status = b3.selectbox("Set status", TASK_STATUSES, index=TASK_STATUSES.index(r.status),
                                      key=f"sel_{r.task_id}", label_visibility="collapsed")
            if new_status != r.status:
                db.update_task_status(r.task_id, new_status)
                st.rerun()

    done = mine[mine["status"] == "Complete"]
    if not done.empty:
        with st.expander(f"Completed operations ({len(done)})"):
            for r in done.itertuples(index=False):
                c1, c2 = st.columns([3, 1])
                c1.write(f"{r.job_id} - {r.part_name} - Op {r.sequence} ({r.machine_id})")
                if c2.button("Reopen", key=f"reopen_{r.task_id}"):
                    db.update_task_status(r.task_id, "In Progress")
                    st.rerun()


def view_admin(db: ShopDB, jobs_rep: pd.DataFrame, machines: pd.DataFrame, tasks: pd.DataFrame) -> None:
    st.title("Admin / Re-Prioritize")
    tab_job, tab_mach, tab_new = st.tabs(["Job Override", "Machines", "Add Job"])
    machine_ids = machines["machine_id"].tolist()

    # ---- Job override ---------------------------------------------------- #
    with tab_job:
        # Sorted by Job ID (not priority) so the selection stays put after a priority change.
        labels = {f"{r.job_id} - {r.customer} - {r.part_name}": r.job_id
                  for r in jobs_rep.sort_values("job_id").itertuples()}
        job_id = labels[st.selectbox("Select Job", list(labels), key="admin_job")]
        j = jobs_rep[jobs_rep["job_id"] == job_id].iloc[0]

        if j["advisor_flag"] == "LATE" or str(j["advisor_flag"]).startswith("BLOCKED"):
            red_flag(f"{job_id}: {j['advisor_flag']} - due {j['due_date']:%Y-%m-%d}, "
                     f"est. completion {j['est_completion'] or 'n/a'}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", j["status"])
        c2.metric("Priority", int(j["priority"]))
        c3.metric("Due", f"{j['due_date']:%b %d}")
        c4.metric("Est. Completion", f"{j['est_completion']:%b %d}" if j["est_completion"] else "n/a")

        with st.form(f"job_form_{job_id}"):
            st.markdown("**Priority & due date override**")
            f1, f2 = st.columns(2)
            pri = f1.number_input("Priority (1 = highest, 10 = lowest)", 1, 10, int(j["priority"]))
            due = f2.date_input("Due Date", j["due_date"])
            if st.form_submit_button("Save job changes", type="primary"):
                db.update_job(job_id, pri, due)
                st.toast(f"{job_id} updated - schedule recalculated")
                st.rerun()

        st.markdown("**Routing - reassign machine, operator, hours, or status**")
        jt = tasks[tasks["job_id"] == job_id][
            ["task_id", "sequence", "machine_id", "assigned_operator", "required_hours", "status"]
        ].reset_index(drop=True)
        edited = st.data_editor(
            jt, key=f"routing_{job_id}", hide_index=True, width="stretch",
            disabled=["task_id", "sequence"],
            column_config={
                "task_id": st.column_config.NumberColumn("Task ID"),
                "sequence": st.column_config.NumberColumn("Op / Seq"),
                "machine_id": st.column_config.SelectboxColumn("Machine", options=machine_ids, required=True),
                "assigned_operator": st.column_config.SelectboxColumn("Operator", options=OPERATORS),
                "required_hours": st.column_config.NumberColumn("Required Hrs", min_value=0.0, step=0.5, format="%.1f"),
                "status": st.column_config.SelectboxColumn("Status", options=TASK_STATUSES, required=True),
            },
        )
        if st.button("Save routing changes", type="primary"):
            changed = 0
            for old, new in zip(jt.itertuples(index=False), edited.itertuples(index=False)):
                if (old.machine_id, old.assigned_operator, old.required_hours, old.status) != \
                   (new.machine_id, new.assigned_operator, new.required_hours, new.status):
                    db.update_task(new.task_id, new.machine_id, new.assigned_operator,
                                   new.required_hours, new.status)
                    changed += 1
            st.toast(f"{changed} operation(s) updated" if changed else "No changes to save")
            st.rerun()

        with st.expander("Quick move: send all open ops on one machine to another"):
            used = sorted(jt.loc[jt["status"] != "Complete", "machine_id"].unique())
            if used:
                q1, q2 = st.columns(2)
                src = q1.selectbox("From machine", used, key=f"src_{job_id}")
                dst = q2.selectbox("To machine", [m for m in machine_ids if m != src], key=f"dst_{job_id}")
                if st.button("Move operations"):
                    n = db.move_job_operations(job_id, src, dst)
                    st.toast(f"Moved {n} operation(s) from {src} to {dst}")
                    st.rerun()
            else:
                st.info("This job has no open operations.")

    # ---- Machines -------------------------------------------------------- #
    with tab_mach:
        st.markdown("Edit daily capacity or mark a machine Down / Maintenance. "
                    "Jobs routed through unavailable machines are flagged on the whiteboard.")
        mt = machines[["machine_id", "machine_name", "daily_capacity_hours", "current_status"]]
        edited_m = st.data_editor(
            mt, key="machines_editor", hide_index=True, width="stretch",
            disabled=["machine_id", "machine_name"],
            column_config={
                "machine_id": "Machine ID", "machine_name": "Machine Name",
                "daily_capacity_hours": st.column_config.NumberColumn("Daily Capacity (h)", min_value=0.0,
                                                                     max_value=24.0, step=0.5),
                "current_status": st.column_config.SelectboxColumn("Status", options=MACHINE_STATUSES,
                                                                   required=True),
            },
        )
        if st.button("Save machine changes", type="primary"):
            for old, new in zip(mt.itertuples(index=False), edited_m.itertuples(index=False)):
                if (old.daily_capacity_hours, old.current_status) != (new.daily_capacity_hours, new.current_status):
                    db.update_machine(new.machine_id, new.daily_capacity_hours, new.current_status)
            st.toast("Machines updated")
            st.rerun()

    # ---- Add job --------------------------------------------------------- #
    with tab_new:
        with st.form("new_job", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)
            new_id = a1.text_input("Job ID", placeholder="J-1009")
            cust = a2.text_input("Customer")
            part = a3.text_input("Part Name")
            b1, b2, b3 = st.columns(3)
            qty = b1.number_input("Quantity", min_value=1, value=1000, step=100)
            due = b2.date_input("Due Date", add_workdays(date.today(), 5))
            pri = b3.number_input("Priority (1-10)", 1, 10, 5)
            st.markdown("**Routing** (operations run top to bottom)")
            routing = st.data_editor(
                pd.DataFrame({"machine_id": [machine_ids[0]], "required_hours": [4.0],
                              "assigned_operator": [OPERATORS[0]]}),
                num_rows="dynamic", hide_index=True, width="stretch", key="new_routing",
                column_config={
                    "machine_id": st.column_config.SelectboxColumn("Machine", options=machine_ids, required=True),
                    "required_hours": st.column_config.NumberColumn("Required Hrs", min_value=0.0, step=0.5),
                    "assigned_operator": st.column_config.SelectboxColumn("Operator", options=OPERATORS),
                },
            )
            if st.form_submit_button("Create job", type="primary"):
                ops = routing.dropna(subset=["machine_id", "required_hours"]).to_dict("records")
                if not new_id.strip() or not cust.strip() or not part.strip():
                    st.error("Job ID, Customer, and Part Name are required.")
                elif new_id.strip() in set(jobs_rep["job_id"]):
                    st.error(f"Job {new_id} already exists.")
                elif not ops:
                    st.error("Add at least one routing operation.")
                else:
                    db.add_job({"job_id": new_id.strip(), "customer": cust.strip(), "part_name": part.strip(),
                                "quantity": int(qty), "due_date": due.isoformat(), "priority": int(pri)}, ops)
                    st.toast(f"Job {new_id} created")
                    st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    db = get_db()
    today = date.today()

    with st.sidebar:
        st.header("Shop Floor Scheduler")
        page = st.radio("View", ["Scheduler Whiteboard", "Shop Floor", "Admin / Re-Prioritize"])
        horizon = st.slider("Capacity horizon (working days)", 1, 20, 5,
                            help="Loaded hours = open work due within this window (incl. overdue).")
        st.caption(f"Schedule start: {add_workdays(today, 0):%a %b %d, %Y}")
        st.caption(f"Database: {'in-memory (demo)' if DB_PATH == ':memory:' else DB_PATH}")
        if st.button("Reset demo data"):
            db.reset()
            st.rerun()

    # Read fresh data and rebuild the schedule on every interaction.
    machines, jobs, tasks = db.machines(), db.jobs(), db.tasks()
    sched_start = add_workdays(today, 0)
    sched = build_schedule(jobs, tasks, machines, sched_start)
    jobs_rep = job_completion_report(jobs, sched, today)
    cap_rep = capacity_report(machines, sched, jobs, today, horizon)

    if page == "Scheduler Whiteboard":
        view_whiteboard(machines, jobs_rep, cap_rep, sched, horizon)
    elif page == "Shop Floor":
        view_shop_floor(db, sched, machines)
    else:
        view_admin(db, jobs_rep, machines, tasks)


if __name__ == "__main__":
    main()

"""Store calibration parameters for semantic predicates in the database and apply them in SQL.

A model's P(yes) can rank well and still sit on the wrong scale (phase 6: this is mainly where the open compatible models lose to Jev).
A few human labels fit two numbers a and b per predicate; the calibrated probability is sigmoid(a + b·logit(p)).
The parameters live in an ordinary table and travel with the database, like statistics:

    semantic_calibration(model, predicate, scope, a, b, labels, positives, question, fitted_at)

    model      the backend's model and version (Backend.model)
    predicate  the predicate's key: md5(question || chr(31) || raw JSON of the criteria); the second part is empty without criteria
    scope      where the parameters apply, usually a table name; '' means this predicate uses them on every table

install() creates the table and a few macros:

    predicate_key(instructions, criteria)                    -> VARCHAR
    calibrated(p, for_model, for_predicate, for_scope)       -> DOUBLE  looks up this scope first, then scope '', and returns p unchanged if neither exists
    <prefix>_noul_cal(state, instructions, scope)              = calibrated(<prefix>_noul(...), ...)
    <prefix>_noulc_cal(state, instructions, criteria, scope)   = calibrated(<prefix>_noulc(...), ...)

The lookup is a join the database does itself, so a changed parameter takes effect on the next query. Fit only on human labels:
open models must not be calibrated on Jev's outputs (TypeSafe customer agreement, section 2.3(b)).
"""

from __future__ import annotations

import hashlib
from typing import Any

import duckdb
import numpy as np

TABLE = "semantic_calibration"
CLIP = 1e-3  # probabilities are clipped to [CLIP, 1 - CLIP] before the logit; the SQL macros use the same number


def predicate_key(instructions: str, criteria: str = "") -> str:
    return hashlib.md5((instructions + "\x1f" + criteria).encode()).hexdigest()


def platt(p: Any, y: Any) -> tuple[float, float]:
    """Fit a + b·logit(p) on human labels. Same algorithm as phase 6's recalibration (platt in experiments/database/p6_report.py):
    targets are pulled in slightly as Platt suggests, so the fit does not diverge when all labels are of one class."""
    from scipy.optimize import minimize
    from scipy.special import logit

    p, y = np.asarray(p, float), np.asarray(y)
    x, n1 = logit(np.clip(p, CLIP, 1 - CLIP)), (y == 1).sum()
    target = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (len(y) - n1 + 2))
    a, b = minimize(lambda w: (np.logaddexp(0, w[0] + w[1] * x) - target * (w[0] + w[1] * x)).sum(), [0.0, 1.0], method="BFGS").x
    return float(a), float(b)


def apply(p: Any, a: float, b: float) -> np.ndarray:
    q = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return 1 / (1 + np.exp(-(a + b * np.log(q / (1 - q)))))


def install(con: duckdb.DuckDBPyConnection, prefix: str, model: str, table: str = TABLE) -> None:
    """Create the parameter table (left as is if it exists) and the macros. Call register(con, prefix, backend) first; model is backend.model."""
    con.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
        model VARCHAR, predicate VARCHAR, scope VARCHAR, a DOUBLE, b DOUBLE, labels INTEGER, positives INTEGER,
        question VARCHAR, fitted_at TIMESTAMP, PRIMARY KEY (model, predicate, scope))""")
    con.execute("CREATE OR REPLACE MACRO predicate_key(instructions, criteria) AS md5(instructions || chr(31) || criteria)")
    # Each of the four expressions passed in appears exactly once, in the innermost SELECT that has no FROM:
    # - so column names inside them can only bind to the caller's query. Written directly into the parameter lookup, a column of the
    #   caller's table that happens to share a name with a parameter-table column (e.g. question) would bind to the parameter table,
    #   the condition would always be true, every row would get the same parameters, and nothing would raise;
    # - and a model call passed in from outside is not expanded into several copies.
    # Parameters come from scalar subqueries (matching scope first, then scope ''), so the question may be a per-row column: each predicate in a query gets its own parameters.
    # least / greatest skip NULL, so rows without an answer (p is NULL) must be kept NULL explicitly; otherwise they would become 1 - CLIP
    con.execute(f"""CREATE OR REPLACE MACRO calibrated(p, for_model, for_predicate, for_scope) AS (
        SELECT CASE WHEN y.p0 IS NULL THEN NULL WHEN y.ab IS NULL THEN y.p0
                    ELSE 1 / (1 + exp(-(y.ab.a + y.ab.b * ln(greatest(least(y.p0, 1 - {CLIP}), {CLIP}) / (1 - greatest(least(y.p0, 1 - {CLIP}), {CLIP})))))) END
        FROM (SELECT x.p0, (SELECT arg_max(struct_pack(a := c.a, b := c.b), (c.scope = x.s0)::INT) FROM {table} c
                            WHERE c.model = x.m0 AND c.predicate = x.k0 AND c.scope IN (x.s0, '')) AS ab
              FROM (SELECT p AS p0, for_model AS m0, for_predicate AS k0, for_scope AS s0) x) y)""")
    name = model.replace("'", "''")
    con.execute(f"""CREATE OR REPLACE MACRO {prefix}_noul_cal(state, instructions, scope) AS
        calibrated({prefix}_noul(state, instructions), '{name}', predicate_key(instructions, ''), scope)""")
    con.execute(f"""CREATE OR REPLACE MACRO {prefix}_noulc_cal(state, instructions, criteria, scope) AS
        calibrated({prefix}_noulc(state, instructions, criteria), '{name}', predicate_key(instructions, criteria), scope)""")


def store(con: duckdb.DuckDBPyConnection, model: str, instructions: str, criteria: str, scope: str, a: float, b: float,
          labels: int, positives: int, table: str = TABLE) -> None:
    con.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, now()::TIMESTAMP)",
                [model, predicate_key(instructions, criteria), scope, a, b, labels, positives, instructions])


def fit(con: duckdb.DuckDBPyConnection, model: str, instructions: str, criteria: str, scope: str, p: Any, y: Any, table: str = TABLE) -> tuple[float, float]:
    """Fit one predicate's parameters on human-labelled rows (p is the model's raw probability, y the 0/1 label) and store them. Rows without an answer (p is NULL) are skipped."""
    p, y = np.asarray(p, float), np.asarray(y)
    keep = ~np.isnan(p)
    a, b = platt(p[keep], y[keep])
    store(con, model, instructions, criteria, scope, a, b, int(keep.sum()), int((y[keep] == 1).sum()), table)
    return a, b

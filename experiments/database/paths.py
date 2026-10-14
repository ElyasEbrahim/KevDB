"""Folders used by this experiment. Paths are computed from this file's location, independent of the working directory.

The cache (.jevcache.sqlite) and .env sit at the repository root like in the other experiments, so scripts still run from the root: python experiments/database/p1_movie.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Third-party data, downloaded by experiments/database/fetch.sh, not committed
DATA = HERE / "data"
# Run outputs, under out/ like the other experiments, not committed
OUT = ROOT / "out" / "database"

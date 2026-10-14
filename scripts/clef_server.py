"""Serve Cloudflare's Clef-flash as a local /v1/systemone endpoint (used from phase 6 on, started by scripts/open_models.sh).

Loading the model, encoding requests and computing probabilities all use joint_schema_model.py from its own repository: `systemone()` takes
Jev's request body and returns Jev's response body. This file only adds the HTTP layer and changes no step.
Run it with the Python of the open_models/clef virtual environment, not the project's .venv.

    open_models/clef/bin/python scripts/clef_server.py <weights folder> --port 8012 [--device mps] [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--port", type=int, default=8012)
parser.add_argument("--device", default="mps")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
args = parser.parse_args()

sys.path.insert(0, args.path)  # joint_schema_model.py lives in the same folder as the weights
from joint_schema_model import load_release_model, systemone  # noqa: E402

started = time.perf_counter()
model, processor = load_release_model(args.path, device=args.device, dtype=getattr(torch, args.dtype))
print(f"serving {args.path} on {args.device} ({args.dtype}), loaded in {time.perf_counter() - started:.0f}s", flush=True)

app = FastAPI()
one_at_a_time = threading.Lock()  # one GPU computes one request at a time


@app.get("/v1/models")
def models() -> dict:
    return {"models": [{"name": "clef-flash", "run": args.path, "device": args.device, "dtype": args.dtype}]}


@app.post("/v1/systemone")
async def answer(request: Request) -> dict:
    body = await request.json()
    try:
        with one_at_a_time:
            return systemone(model, processor, body)
    except ValueError as e:  # systemone raises ValueError for an invalid request body
        raise HTTPException(422, str(e)) from e


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

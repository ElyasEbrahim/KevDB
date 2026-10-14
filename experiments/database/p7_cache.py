"""Phase 7, part B: Kev's server caches the text it has read; how much cheaper is asking the same rows another question? Criteria: experiments/database/protocol/p7.txt.

One run = one dataset × one arm × one repetition, against a freshly started server whose settings match the arm (experiments/database/p7_run.sh starts and stops it):

    off      KEV_PREFIX_CACHE=0, no caching, every question re-reads the text (control)
    table    cache large enough for every row of this dataset
    default  the factory setting: 4 prefixes

Steps: warm up on made-up rows, then ask the same rows each question once in a fixed order (concurrency 1, so request times include no queueing). After each block of rows,
the server's own hit / miss counts and the server process's memory are read; these readings are not timed. The off arm finally asks the questions packed into one request, as a reference.
Answers are stored as they come and not looked at until everything is collected.

    python experiments/database/p7_cache.py env movie table      # environment variables for this arm, used by p7_run.sh to start the server
    python experiments/database/p7_cache.py env budget           # environment variables for the SQL step's "cache up to the memory budget"
    python experiments/database/p7_cache.py run movie table 1    # the server must already be running with the settings above
    python experiments/database/p7_cache.py check                # are all files there, are there failed requests? Counting only
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

from kevdb.client import PROVIDERS

from kevdb import OPEN_MODELS, open_backend
from p7_data import ARMS, CACHE_SETS, MAIN_SETS, MODEL, OUT as ROOT_OUT, REPS, SWEEP_SETS, cache_set, server_settings

OUT = ROOT_OUT / "cache"
BASE = PROVIDERS[OPEN_MODELS[MODEL].provider].base_url


def plan() -> list[tuple[str, str, int]]:
    """Every run, in execution order. Main data: REPS passes of off and table, off first on odd passes and table first on even ones, so drift in the machine's state does not always hit the same arm;
    then one pass of default. The long-document strata: one pass each of off and table."""
    runs = [(s, arm, rep) for rep in range(1, REPS + 1) for s in MAIN_SETS for arm in (("off", "table") if rep % 2 else ("table", "off"))]
    return runs + [(s, "default", 1) for s in MAIN_SETS] + [(s, arm, 1) for s in SWEEP_SETS for arm in ("off", "table")]


def stem(name: str, arm: str, rep: int) -> str:
    return f"{name}.{arm}.r{rep}"


def server() -> dict:
    with urllib.request.urlopen(f"{BASE}/models", timeout=30) as resp:
        return json.load(resp)["models"][0]


def footprint(pid: int) -> int | None:
    """Physical memory actually used by the server process (bytes), including what MLX keeps in GPU memory. ps's RSS excludes that, so the system's footprint tool is used."""
    out = subprocess.run(["footprint", "-f", "bytes", str(pid)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "phys_footprint:" in line:
            return int(line.split(":")[1].split()[0])
    return None


def listener() -> int:
    port = BASE.rsplit(":", 1)[1].split("/")[0]
    return int(subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()[0])


def available_memory() -> int:
    """Free plus immediately reclaimable memory of the system (bytes), recorded only: with too little at the start of a run, timings may suffer from paging."""
    pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in (line.split(":") for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines()[1:] if ":" in line)}
    return 16384 * (pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0) + pages.get("Pages purgeable", 0))


def ask(backend, items: list[tuple]) -> tuple[list[dict], float, int]:
    """Ask one block. Failed requests are re-asked up to 3 rounds within the block, and that time counts. Returns (answers, seconds, number of re-asked requests)."""
    started = time.perf_counter()
    answers = backend.ask_many(items)
    retried = 0
    for _ in range(3):
        failed = [i for i, a in enumerate(answers) if "_error" in a]
        if not failed:
            break
        retried += len(failed)
        for i, a in zip(failed, backend.ask_many([items[i] for i in failed])):
            answers[i] = a
    return answers, time.perf_counter() - started, retried


def one_pass(backend, pid: int, rows: list[dict], questions: dict[str, dict], block: int, path) -> dict:
    """Ask the same rows once (questions holding several questions means packing them). After each block, read the server's counters and memory."""
    before, blocks, seconds, retried, latencies = server()["prefix_cache"], [], 0.0, 0, []
    with path.open("w") as f:
        for start in range(0, len(rows), block):
            chunk = rows[start:start + block]
            answers, dt, again = ask(backend, [(r["state"], questions) for r in chunk])
            seconds, retried = seconds + dt, retried + again
            for r, a in zip(chunk, answers):
                ok = "_error" not in a
                latencies += [a["_latency"]] if ok else []
                f.write(json.dumps({"id": r["id"], "answers": a["answers"] if ok else None, "tokens": a["usage"]["input_tokens"] if ok else None,
                                    "latency": a["_latency"] if ok else None, "error": a.get("_error")}) + "\n")
            now = server()["prefix_cache"]
            blocks.append({"rows": len(chunk), "seconds": round(dt, 3), "tokens": sum(a["usage"]["input_tokens"] for a in answers if "_error" not in a),
                           "errors": sum("_error" in a for a in answers), "cached_states": now["cached_states"], "footprint": footprint(pid)})
    after, ordered = server()["prefix_cache"], sorted(latencies)
    return {"questions": list(questions), "rows": len(rows), "seconds": round(seconds, 3), "errors": sum(b["errors"] for b in blocks), "retried": retried,
            "hits": after["hits"] - before["hits"], "misses": after["misses"] - before["misses"], "oom_retries": after["oom_retries"] - before["oom_retries"],
            "latency_p50": ordered[len(ordered) // 2] if ordered else None, "latency_p95": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)] if ordered else None,
            "blocks": blocks}


def run(name: str, arm: str, rep: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / f"{stem(name, arm, rep)}.json"
    if done.exists():
        print(f"{stem(name, arm, rep)}: results exist, skipped")
        return
    rows, questions, block, warm = cache_set(name)
    card = server()
    # The server's settings must match this arm, otherwise stop: numbers collected against the wrong settings mean nothing
    wanted = {"off": 0, "default": 4}.get(arm, len(rows) + len(warm))
    if card["prefix_cache"]["size"] != wanted or card["prefix_cache"]["cached_states"] != 0:
        sys.exit(f"the server's cache setting is {card['prefix_cache']}; the {arm} arm needs size={wanted}, freshly started with nothing cached yet")
    pid = listener()
    backend = open_backend(MODEL, concurrency=1, cache_path=None)
    record = {"set": name, "arm": arm, "rep": rep, "model": backend.model, "backend": card["backend"], "dtype": card["dtype"], "settings": server_settings(arm, len(rows), len(warm)),
              "prefix_cache": card["prefix_cache"], "rows": len(rows), "block": block, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "available_memory": available_memory(), "footprint_loaded": footprint(pid)}
    # Warm-up: ask every question once on made-up rows, so the first-time cost of each shape stays out of the timing. These rows are unrelated to the data and their answers are discarded
    warmed = [a for q in questions.values() for a in backend.ask_many([(r["state"], {"q": q}) for r in warm])]
    record["warmup"] = {"requests": len(warmed), "errors": sum("_error" in a for a in warmed), "tokens": sum(a["usage"]["input_tokens"] for a in warmed if "_error" not in a) // len(questions),
                        "cached_states": server()["prefix_cache"]["cached_states"], "footprint": footprint(pid)}
    record["passes"] = [one_pass(backend, pid, rows, {"q": q}, block, OUT / f"{stem(name, arm, rep)}.{i}.jsonl") | {"question": q_name}
                        for i, (q_name, q) in enumerate(questions.items(), 1)]
    if arm == "off":
        record["packed"] = one_pass(backend, pid, rows, questions, block, OUT / f"{stem(name, arm, rep)}.packed.jsonl")
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    backend.close()
    done.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(f"{stem(name, arm, rep)}: " + ", ".join(f"question {i} {p['seconds']:.1f} s ({p['hits']} hits, {p['errors']} failed)" for i, p in enumerate(record["passes"], 1)), flush=True)


def check() -> None:
    missing = []
    print(f"{'run':24s}{'rows':>6s}{'questions':>10s}{'failed':>8s}{'re-asked':>9s}{'OOM retries':>12s}")
    for name, arm, rep in plan():
        path = OUT / f"{stem(name, arm, rep)}.json"
        if not path.exists():
            missing.append(stem(name, arm, rep))
            continue
        r = json.loads(path.read_text())
        every = r["passes"] + ([r["packed"]] if "packed" in r else [])
        print(f"{stem(name, arm, rep):24s}{r['rows']:>8d}{len(r['passes']):>8d}{sum(p['errors'] for p in every):>12d}{sum(p['retried'] for p in every):>10d}{sum(p['oom_retries'] for p in every):>14d}")
    if missing:
        sys.exit("missing: " + "; ".join(missing))
    print("all files complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["env", "budget"]:
        print(" ".join(f"{k}={v}" for k, v in server_settings("budget", 0, 0).items()))
    elif args[:1] == ["env"] and len(args) == 3 and args[1] in CACHE_SETS and args[2] in ARMS:
        rows, _, _, warm = cache_set(args[1])
        print(" ".join(f"{k}={v}" for k, v in server_settings(args[2], len(rows), len(warm)).items()))
    elif args[:1] == ["run"] and len(args) == 4 and args[1] in CACHE_SETS and args[2] in ARMS:
        run(args[1], args[2], int(args[3]))
    elif args == ["plan"]:
        print("\n".join(f"{s} {arm} {rep}" for s, arm, rep in plan()))
    elif args == ["check"]:
        check()
    else:
        sys.exit(__doc__)

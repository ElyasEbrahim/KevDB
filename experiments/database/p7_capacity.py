"""Phase 7 capacity probe: how many short prefixes can a local model's text cache safely hold on this machine? Uses made-up rows only and never touches evaluation data.

The owner decided on 2026-10-12 to "use as much memory as possible", so the cache's memory budget is no longer a fixed number but measured per model before collection:

    KEV_PREFIX_CACHE=100000 KEV_PREFIX_MAX_TOKENS=4000000 ./scripts/open_models.sh serve kev-4b     # in another terminal
    P7_MODEL=kev-4b python experiments/database/p7_capacity.py

Method: feed made-up reviews into the cache in blocks of 25; after each block read the server's memory, the system's available memory and swap, and ask the earliest 20
another question (they should all hit the cache) to see whether hits get slower. Stop at the first of the following; the last block before stopping counts as "still healthy":

    system memory available below 4 GB; the server process above 75% of physical memory (Metal's recommended working-set limit); swap grown by more than 1 GB since the start;
    median hit latency above 1.25x that of the first block; the server reported running out of memory; more than 4,000 entries (the largest table has 3,949 rows, more is useless).

Budget: when memory stopped the probe, 90% of the cache's size at the last healthy block becomes the byte budget, and the entry count follows from the mean size per entry, rounded down to a multiple of 50;
if 4,000+ entries fit without hitting any condition, memory is not the limit and the budget is the size and count at that point (rounded to 50).
The result goes to out/database/p7/<model>/capacity.json; copy its budget into CAPACITY in experiments/database/p7_data.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import p7_cache as C
from kevdb import open_backend
from p7_data import MAX_TOKENS, MODEL, MOVIE_QUESTIONS, RESULTS, WARM_SHORT, WHOLE_TABLES, made_reviews

BLOCK, PROBE_ROWS = 25, 20
FLOOR = 4 * 2**30
LIMIT = (max(WHOLE_TABLES.values()) + WARM_SHORT + 2 * BLOCK) // BLOCK * BLOCK
MARGIN, SLOWER, SWAP_GROWTH, SHARE_OF_RAM = 0.9, 1.25, 2**30, 0.75
OUT = RESULTS / "p7" / MODEL


def swap_used() -> int:
    text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    value = text.split("used =")[1].split()[0]
    return int(float(value[:-1]) * {"M": 2**20, "G": 2**30, "K": 2**10}[value[-1]])


def main() -> None:
    card = C.server()
    if card["prefix_cache"]["size"] < LIMIT or card["prefix_cache"]["max_tokens"] < MAX_TOKENS or card["prefix_cache"]["cached_states"]:
        sys.exit(f"the server must be freshly started with a large enough cache (KEV_PREFIX_CACHE=100000 KEV_PREFIX_MAX_TOKENS={MAX_TOKENS}); it is now {card['prefix_cache']}")
    pid, ram = C.listener(), int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    backend = open_backend(MODEL, concurrency=1, cache_path=None)
    rows = made_reviews("capacity", LIMIT)
    read, hit = {"q": MOVIE_QUESTIONS["clearly"]}, {"q": MOVIE_QUESTIONS["positive"]}
    start = {"footprint": C.footprint(pid), "available": C.available_memory(), "swap": swap_used()}
    blocks, reason, baseline = [], None, None
    for at in range(0, LIMIT, BLOCK):
        answers = backend.ask_many([(r["state"], read) for r in rows[at:at + BLOCK]])
        again = backend.ask_many([(r["state"], hit) for r in rows[:PROBE_ROWS]])
        latency = sorted(a["_latency"] for a in again if "_error" not in a)
        now = C.server()["prefix_cache"]
        block = {"states": at + BLOCK, "cached_states": now["cached_states"], "tokens": sum(a["usage"]["input_tokens"] for a in answers if "_error" not in a),
                 "errors": sum("_error" in a for a in answers + again), "footprint": C.footprint(pid), "available": C.available_memory(), "swap": swap_used(),
                 "hit_latency_p50": latency[len(latency) // 2] if latency else None, "oom_retries": now["oom_retries"]}
        baseline = baseline or block["hit_latency_p50"]
        if block["errors"] or block["oom_retries"] or block["cached_states"] != block["states"]:
            reason = "the server reported an error or ran out of memory, or the cached entry count does not match"
        elif block["available"] < FLOOR:
            reason = f"system memory available below {FLOOR / 2**30:.0f} GB"
        elif block["footprint"] > SHARE_OF_RAM * ram:
            reason = f"server process above {SHARE_OF_RAM:.0%} of physical memory"
        elif block["swap"] - start["swap"] > SWAP_GROWTH:
            reason = "swap grew by more than 1 GB since the start"
        elif block["hit_latency_p50"] > SLOWER * baseline:
            reason = f"hit latency above {SLOWER}x that of the first block"
        blocks.append(block | {"healthy": reason is None})
        print(f"{block['states']:>5d} entries: server {block['footprint'] / 2**30:5.1f} GB, system available {block['available'] / 2**30:5.1f} GB, swap {block['swap'] / 2**30:4.1f} GB, "
              f"median hit {block['hit_latency_p50'] * 1000:.0f} ms" + (f"  ← stop: {reason}" if reason else ""), flush=True)
        if reason:
            break
    backend.close()
    healthy = [b for b in blocks if b["healthy"]]
    if not healthy:
        sys.exit("the very first block is unhealthy; no budget can be set")
    last = healthy[-1]
    used = last["footprint"] - start["footprint"]
    per_state = used / last["states"]
    if reason:
        budget_bytes = int(MARGIN * used)
        budget = {"states": int(budget_bytes / per_state // 50 * 50), "bytes": budget_bytes}
    else:
        budget = {"states": last["states"] // 50 * 50, "bytes": int(used)}
    result = {"model": card["run"], "backend": card["backend"], "dtype": card["dtype"], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ram": ram, "start": start,
              "stopped_because": reason or f"{LIMIT} entries fit without hitting any stop condition: memory is not the limit", "last_healthy": last, "bytes_per_state": per_state, "budget": budget, "blocks": blocks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capacity.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{MODEL}: the last healthy block holds {last['states']} entries, the cache uses {used / 2**30:.1f} GB ({per_state / 2**20:.1f} MB each). {result['stopped_because']}.")
    print(f"budget: {budget['states']} entries, {budget['bytes'] / 2**30:.1f} GB  →  CAPACITY[\"{MODEL}\"] = {json.dumps(budget)}")


if __name__ == "__main__":
    main()

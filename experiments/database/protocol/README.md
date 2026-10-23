# Pre-registered criteria

These files are the original records of the experiments: for each phase, the primary analysis, success criteria and
stopping rules written before the run, followed by notes on deviations appended afterwards.

| File | Content |
| --- | --- |
| `p0.txt` | rule for choosing the upper-tier LLM |
| `p3.txt` | primary analysis of semantic aggregation |
| `p4.txt` | selectivity, the restaurant pre-check, the packing comparison, ordering with Jev only |
| `p5.txt` | follow-up with a stronger upper-tier model |
| `p6.txt` | can open, Jev-compatible models replace Jev as the default backend? (from 2026-10-09) |
| `p6_plan.md` | phase 6 execution plan: steps, acceptance checks, rules for mishaps. `p6.txt` prevails on criteria |
| `p7.txt` | does a local decision model inside DuckDB offer anything a cloud API cannot? (from 2026-10-11) |
| `p7_plan.md` | phase 7 execution plan. `p7.txt` prevails on criteria |
| `p8.txt` | semantic joins: a left record's candidates pair by pair or in one request? (2026-10-13) |
| `p9.txt` | does phase 8's packed form carry over to joins other than record matching? (2026-10-13) |
| `p10.txt` | does fine-tuning Kev on public record-matching data reach Jev's zero-shot level on unseen matching data? (written 2026-10-13, frozen 2026-10-15, run 2026-10-16 to 2026-10-20; the execution record and every deviation are appended at its end) |
| `p10_plan.md` | phase 10 execution plan. `p10.txt` prevails on criteria |
| `p11.txt` | why did fine-tuning not improve Kev-4B on dblp-acm in phase 10? Three arms, each adding one ingredient (written and frozen 2026-10-20, run 2026-10-20 to 2026-10-21; the execution record is appended at its end) |
| `p12.txt` | with how many of their own labels does Kev-4B reach Jev's zero-shot level on a matching dataset it has not seen? (written 2026-10-20, frozen and run 2026-10-21 to 2026-10-22; the execution record is appended at its end) |
| `p13.txt` | are phase 10's close verdicts stable when only the training seed changes? (arms chosen by a rule on 2026-10-20, frozen and run 2026-10-22 to 2026-10-23; the execution record is appended at its end) |
| `p14.txt` | is Kev-0.8B's drop on movie reviews in phase 10 a property of seed 1 or of training without padding? (written, frozen and run 2026-10-23; a false start before the rehearsal and the execution record are appended at its end) |

## Paths and checksums

Each protocol records the paths and SHA-256 checksums of the scripts at the time it was written. Those records are
historical and are not updated:

- `p0.txt`–`p5.txt` refer to scripts under `exp/…`, the folder used before the experiments moved to `database/`.
- `p6.txt`–`p10.txt` refer to `database/…` and `src/jevsql/…`.
- `p11.txt` and later were written after the move and refer to the current paths; their checksums are those of the
  files in this repository at the time they were frozen.
- For the public release the experiments moved again (`database/` → `experiments/database/`, `src/jevdata/` and
  `src/jevsql/` → `src/kevdb/`, `database/open_models.sh` → `scripts/open_models.sh`), and every script's comments,
  docstrings and printed messages were translated into English. No code, question, prompt or threshold changed, but the
  current files no longer match any recorded checksum. The original files are preserved outside the repository.

## Edits made for the public release

The protocols are otherwise unchanged. The only edits:

- `p5.txt`: the upper-tier model's name is shortened to "Opus 5.5".
- `p7.txt`: two references to the README section of phase 7 now use its English title.
- `p6_plan.md`, `p7_plan.md` and `p10_plan.md` were written in Chinese and are English translations of the originals.

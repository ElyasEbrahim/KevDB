# Batch pipeline and CLI

Besides SQL, `kevdb` ships a command-line pipeline for labelling a data file in one go: every row becomes a state, all
questions about the row go into one request, answers are flattened into columns, and rows with low confidence are routed
to review.

```text
data file (.jsonl / .json / .csv)
  → keep only the task's state_fields as the row's state
  → one request per row with all questions (concurrent, backoff on 429/529, SQLite cache so runs resume)
  → answers flattened into columns: <question>, <question>_conf (for noul: <question>_p)
  → rows below review_below get _review=True and also go to a separate review file
```

## Question types

| Type | Asks | Returns |
| --- | --- | --- |
| `noul` | yes / no | P(yes) |
| `choice` | pick one of the given options | the option, each option's probability, a confidence |
| `score` | rate on 2–10 levels | probability-weighted score and a confidence |

## A task file

```toml
name = "posts_triage"
input = "experiments/posts_triage/sample_posts.jsonl"
id_field = "id"
state_fields = ["text", "lang", "video"]
provider = "typesafe"
model = "jev-1.13.0"
concurrency = 16

[questions.content_type]
type = "choice"
instructions = "What kind of post is `text`? Pick the single best fit."
review_below = 0.6

[questions.content_type.criteria]
demo = "The author shows something they built or ran ..."
tutorial = "Step-by-step instructions ..."
other = "Too short or too vague to classify ..."

[questions.first_hand]
type = "noul"
instructions = "Does the author of `text` describe something they personally built, ran, or measured?"
review_below = 0.6
```

The complete example is [experiments/posts_triage/posts_triage.toml](../experiments/posts_triage/posts_triage.toml). To
write a new task, copy it and change `input`, `state_fields` and `[questions.*]`.

Tips that held up in the experiments:

- Write instructions in English with exact conditions, and put edge cases into `criteria`.
- One question judges one thing. Counting, comparing numbers and date arithmetic belong in code, or bucket values first.
- Send only the fields the decision needs: an irrelevant, large state lowers accuracy.

## Commands

```bash
kevdb run TASK.toml --mock                 # offline fake backend: no key, meaningless answers, checks the pipeline
kevdb run TASK.toml                        # real run with the task's provider and model
kevdb run TASK.toml --input other.json --records-path items --limit 100 --tag try1
kevdb run TASK.toml --llm deepseek-v4.1-flash --reasoning none     # the same questions answered by a generative LLM
kevdb sample TASK.toml --pred out/TASK.FILE.jsonl --class-field content_type --out out/gold_todo.jsonl
kevdb eval TASK.toml --pred out/TASK.FILE.jsonl --gold gold.jsonl
kevdb ping                                 # check that a key works
```

Outputs go to `out/`: `<task>.<input file>.jsonl`, `.csv` and `.review.jsonl`.

### Measuring accuracy and setting thresholds

1. `kevdb sample` draws rows uniformly at random, plus extra rows for rare predicted classes. The file contains only the
   original fields, never the model's answers, so labelling stays blind.
2. Label it into a gold file: one line per row with `id` and the correct answer per question. Rows that genuinely fit two
   classes can list `<question>_alt: [...]`; leave a question out when unsure and it is not scored.
3. `kevdb eval` reports accuracy per question, a calibration table (confidence bin vs actual accuracy), a threshold sweep
   (how many rows pass automatically at each `review_below` and how accurate they are) and the most common errors.

## Providers and keys

| Provider | `.env` variable | Used for |
| --- | --- | --- |
| TypeSafe official API | `TYPESAFE_API_KEY` | Jev, `jev-1.13.0` |
| OpenCode Zen | `OPENCODE_API_KEY` | comparison only: LLM baselines and the route comparison |
| OpenRouter | `OPENROUTER_API_KEY` | LLMs in the database experiments, provider pinned per route |

Copy `.env.example` to `.env` and fill in only what you need; nothing is required for the tests or `--mock`. When both
Jev keys are set, the official API is the default: the third-party route rate-limits (about 1,000 requests per window)
and doubles single-request latency.

The CLI targets the hosted APIs. To label data with a local open model, use the SQL functions with
`open_backend(...)` (see [sql-functions.md](sql-functions.md)): its answers are cached under the pinned weight version,
which keeps Kev-4B, Kev-9B and Kev-0.8B apart.

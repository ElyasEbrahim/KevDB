# Reliable aggregates from a few human labels: `kevdb.allocation`

A cheap model can answer every row of a large table, but its answers are biased: on movie reviews Jev is right 91% of
the time, yet the share of positive reviews it implies is 5.9 points too low, and per movie a quarter of the shares are
off by more than 10 points. `kevdb.allocation` turns the cheap answers plus a few human labels into estimates that are
unbiased against the human labels, with 95% intervals. It also chooses which rows get a label, from the query: the
rows its answer depends on most.

## Use

```python
import duckdb
from kevdb import allocation, open_backend, register

con = duckdb.connect()
register(con, "kev", open_backend("kev-4b"))          # any backend works; this one runs locally
con.execute("""CREATE TABLE scored AS
               SELECT id, movie, review, kev_noul(json_object('review', review), 'Is the movie review in `review` positive?') AS p
               FROM reviews""")

plan = allocation.plan(con, "scored", key="id", p="p", group="movie", budget=2000)
plan.rows()                       # the rows a person should label: key, p, inclusion_probability
labels = {...}                    # key -> 0 or 1, for every row in plan.rows()
plan.estimate(labels)             # per movie: estimate, low, high, labelled, rows
```

Each output is a share, `sum(weight × label) / sum(weight)` over the rows of its group:

| Argument | What it says about the query |
| --- | --- |
| `group` | the GROUP BY column; a list column when a row counts in several groups (a review counted in every genre of its movie, through a join) |
| `weight` | a non-negative weight per row: a measure (box office), or how many output rows a join makes of the row |
| `h` | the 95% half-width each output should reach, as one number or per group; equal for all outputs by default |
| `budget` | the expected number of labels |

`key`, `p`, `group` and `weight` are SQL expressions on the table, so `p` can be the model call itself. Leave rows that
are not part of the query out of the table: a row of weight 0 still takes a share of the uniform part of the budget.
[experiments/database/p16_demo.py](../experiments/database/p16_demo.py) runs all three kinds (per movie, per genre
through a join, weighted by box office) on Jev's cached answers.

## What it guarantees

- **No bias from the model, as long as the labels are the truth.** The labels correct the model's answers on every row,
  each weighted by its inclusion probability, so a biased or miscalibrated model costs precision, never validity.
- **Every chosen row must be labelled, and no other label may be added.** A missing label, or labels chosen by hand,
  breaks the guarantee; `estimate` refuses a missing one.
- **95% intervals from the labels themselves**, with Student's t on the number of labels behind each output. An output
  with too few labels gets [0, 1], and a point only when every row of it was labelled.
- **A second model's answers are not labels.** In phase 16, intervals for the share of positive reviews built that way
  covered the second model's own share 94–95% of the time and the human truth 0–0.1%: when Jev is wrong, other models
  are wrong 64–85% of the time.

## How it picks rows, and what that saves

A row's chance of being labelled is proportional to its influence on the outputs (its weight over its group's total,
summed over the groups it counts in) times the model's uncertainty `sqrt(p(1 - p))`. A fifth of the budget is spread
uniformly, so that no row's chance is negligible where the model is confidently wrong. Small groups are therefore
labelled far more densely than large ones, and rows a join repeats many times are labelled more often.

Measured in [phase 16](../experiments/database/README.md#phase-16-can-the-query-decide-which-rows-get-the-expensive-answer)
on 15,388 movie reviews and 3,949 restaurant sentences: the labels other designs need for the same error as this one,
whose 95% intervals covered the truth as required on every query.

| Query | Uniform sampling | The model's confidence alone |
| --- | --- | --- |
| Share per movie (172 groups of 4–503 reviews) | 11.8× | 9.4× |
| Share per genre, through a join | 8.7× | 6.6× |
| Box-office-weighted mean of the movie shares | 11.5× | 8.6× |
| A join repeating reviews with a heavy-tailed count (simulated repetitions) | 2.1–29× | 1.7–25× |
| The overall share of positive reviews | 1.2× | 1.0× |
| 21 restaurant predicates over the same sentences, many rare | 1.0× | 1.0× |

Spending 30% of the budget first on a human pilot to find the model's errors did not help (this design needs
0.95–1.10× the labels of that one), so there is no pilot stage.

## Limits

- Shares only. A count is a share times the group's known number of rows.
- Small groups need most of their rows labelled to get a narrow interval; with a few hundred labels and many tiny groups,
  many intervals stay wide.
- Heavy weights or heavy reuse need more labels before the intervals hold: in phase 16, 162 labels were not enough for
  the heavy-tailed join on 5,001 rows.
- The labels must come from the person or process that defines the truth; labels from a model inherit its errors.

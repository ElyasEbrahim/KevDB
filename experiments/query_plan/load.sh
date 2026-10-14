#!/usr/bin/env bash
# Start the database, create the tables, load the CSV files, index the foreign keys and gather statistics. Skips everything if the data is already loaded.
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d --wait
psql() { docker exec -i jev-query-plan-db psql -U job -d imdb -v ON_ERROR_STOP=1 -q "$@"; }

if [ "$(psql -tAc "select to_regclass('public.title') is not null")" = "t" ] \
   && [ "$(psql -tAc "select count(*) > 0 from title")" = "t" ]; then
  echo "data already loaded, skipping."
else
  psql < data/job-queries/schema.sql
  for csv in data/imdb/*.csv; do
    table="$(basename "$csv" .csv)"
    echo "loading $table"
    # IMDb's CSV files escape quotes with a backslash
    psql -c "COPY $table FROM '/data/imdb/$table.csv' WITH (FORMAT csv, ESCAPE '\\')"
  done
  psql < data/job-queries/fkindexes.sql
  psql -c "VACUUM ANALYZE"
fi

psql -tAc "select 'rows in title: ' || count(*) from title"
psql -tAc "select 'database size: ' || pg_size_pretty(pg_database_size('imdb'))"

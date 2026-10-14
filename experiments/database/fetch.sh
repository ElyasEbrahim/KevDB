#!/usr/bin/env bash
# Download every third-party dataset the experiments use, plus two reference papers. Safe to re-run; existing files are not downloaded again.
#   data/sembench    the SemBench benchmark (Apache-2.0 / MIT): data, queries, ground truth, published results of each system. Only files/ and src/
#   data/mams        MAMS-ACSA: restaurant reviews, each sentence labelled by humans with several aspects and their sentiment (Apache-2.0), about 1.4 MB
#   data/goemotions  the raw GoEmotions labels, including each annotator's (Apache-2.0), about 43 MB
#   refs/            the PDFs of the JEVDB and SemBench papers
# Both Git repositories are pinned to a commit so results are reproducible.
set -euo pipefail
cd "$(dirname "$0")"

SEMBENCH_COMMIT=8c2cac89b21c6782239e8e5b03d326d0c029b655
MAMS_COMMIT=cddcdb0f423b3fdb2c76b70744737abed3a00d17

get() {  # get <local path> <URL>. Writes to .part and renames when complete; an interrupted download resumes on the next run
  mkdir -p "$(dirname "$1")"
  [ -s "$1" ] || { curl -sS --fail -L -C - -o "$1.part" "$2" && mv "$1.part" "$1"; }
}

if [ ! -d data/sembench/.git ]; then
  git init -q data/sembench
  git -C data/sembench remote add origin https://github.com/SemBench/SemBench.git
  # docs/ and figures/ are the website and images, most of the repository, and not needed
  git -C data/sembench sparse-checkout set files src
  git -C data/sembench fetch -q --depth 1 --filter=blob:none origin "$SEMBENCH_COMMIT"
  git -C data/sembench checkout -q FETCH_HEAD
fi
movie=data/sembench/files/movie
echo "sembench: movie scenario, $(($(wc -l < $movie/data/sf_2000/Reviews.csv) - 1)) reviews, $(ls $movie/query/natural_language | wc -l | tr -d ' ') queries"

for split in train val test; do
  get "data/mams/$split.xml" "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/$MAMS_COMMIT/data/MAMS-ACSA/raw/$split.xml"
done
echo "mams: $(cat data/mams/*.xml | grep -c '<sentence>') sentences"

for part in 1 2 3; do
  get "data/goemotions/goemotions_$part.csv" "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_$part.csv"
done
echo "goemotions: $(($(cat data/goemotions/*.csv | wc -l) - 3)) labels"

# The source data of SemBench's movie scenario (Kaggle's Rotten Tomatoes reviews, about 1.4 million), the copy the SemBench authors keep on Google Drive, 158 MB.
# The repository only ships the 2,000-row version; larger scales need this.
rt=data/rotten_tomatoes
if [ ! -s $rt/rotten_tomatoes_movie_reviews.csv ]; then
  get $rt/movie.zip "https://drive.usercontent.google.com/download?id=1WkeXN3V5A6h-D_oeWvv2G_xs2PgzcQu0&export=download&confirm=t"
  unzip -q -o -j $rt/movie.zip rotten_tomatoes_movies.csv rotten_tomatoes_movie_reviews.csv -d $rt
  rm $rt/movie.zip
fi
echo "rotten_tomatoes: $(du -sh $rt | cut -f1 | tr -d ' ')"

# Files on Google Drive and Google Cloud Storage have no version, so checksums confirm it is still the same file
shasum -a 256 -c --quiet <<'EOF'
fce65e53761e807276a16a2f1409da13148450036518323b2b37202142cb322f  data/rotten_tomatoes/rotten_tomatoes_movie_reviews.csv
7a2e0e33d77def8b5da502227d9465a07c4131055c5a4c55d0653e2a490512bf  data/rotten_tomatoes/rotten_tomatoes_movies.csv
cac049036bad5d68d1081f72b65f2cc51e4df82af05e3e22cfa747051cac1af3  data/goemotions/goemotions_1.csv
f699ecc5aa425c1720c1d02475f1e41815244b680bd75b282eb770d2c76cd84d  data/goemotions/goemotions_2.csv
467f1e7191af00f2e76cc7f425885c2dc304bea8aff284b10e8c460d22f2e1af  data/goemotions/goemotions_3.csv
EOF

# Draw a 16,000-row version from the source data with SemBench's own script (needs the .venv at the repository root, see experiments/database/README.md).
# Always use the shipped 2,000-row version and never regenerate it: with another pandas version about 0.5% of the rows differ from the shipped ones,
# and the published results of every system refer to the shipped file. The same goes for the 16,000-row version: it is "drawn the same way", not guaranteed identical row for row to what SemBench used.
big=$movie/data/sf_16000
[ -s $big/Reviews.csv ] || ../../.venv/bin/python data/sembench/src/scenario/movie/preparation/generate_data.py $rt 16000 > /dev/null
echo "sembench sf_16000: $(($(wc -l < $big/Reviews.csv) - 1)) reviews"

# DuckDB's full-text search extension is not bundled and is downloaded on first use; install it here
../../.venv/bin/python -c "import duckdb; duckdb.sql('INSTALL fts')"

get refs/jevdb-2610.02046v1.pdf https://arxiv.org/pdf/2610.02046v1
get refs/sembench-2511.01716v2.pdf https://arxiv.org/pdf/2511.01716v2
echo "refs: $(ls refs/*.pdf | wc -l | tr -d ' ') papers"

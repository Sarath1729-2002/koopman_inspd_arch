#!/bin/bash
# Print benchmark sections whose heading matches a pattern (default: everything) + all pre-registered verdicts.
#   bash sf_cc/show.sh "third|Deep"
cd "$(dirname "$0")/.." || exit 1
P=${1:-.}
awk -v pat="$P" '/^## /{p = ($0 ~ pat) && ($0 !~ /^## Pre-registered/)} p' results_split/BENCH.md
grep -h "^- " results_split/BENCH.md

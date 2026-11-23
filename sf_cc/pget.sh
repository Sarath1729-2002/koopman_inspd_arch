#!/bin/bash
# Parallel resumable download of the rest of a file: N concurrent HTTP range requests, then append in order + sha256.
# Usage: bash sf_cc/pget.sh URL PARTFILE SHA256 [N]   (PARTFILE = what is already downloaded; it is extended in place)
set -uo pipefail
URL=$1; PART=$2; SHA=$3; N=${4:-8}
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
TOTAL=$(curl -sIL "${AUTH[@]}" "$URL" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
[ -n "$TOTAL" ] || { echo "could not read file size"; exit 1; }
HAVE=$(stat -c %s "$PART"); REST=$((TOTAL - HAVE)); echo "total $TOTAL have $HAVE rest $REST in $N pieces"
[ $REST -gt 0 ] || { echo "already complete"; }
CH=$(( (REST + N - 1) / N )); D=$(dirname "$PART")/pieces; mkdir -p "$D"
for ((i = 0; i < N; i++)); do
  a=$((HAVE + i * CH)); b=$((a + CH - 1)); [ $b -ge $TOTAL ] && b=$((TOTAL - 1)); [ $a -gt $b ] && continue
  ( want=$((b - a + 1))
    for t in $(seq 1 30); do
      got=$(stat -c %s "$D/$i" 2>/dev/null || echo 0); [ "$got" -ge "$want" ] && break
      curl -sL "${AUTH[@]}" -r $((a + got))-$b "$URL" >> "$D/$i"; sleep 2
    done ) &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
  sleep 30; echo "$(date +%T) $(( $(cat "$D"/* 2>/dev/null | wc -c) / 1000000 )) MB of $((REST / 1000000)) MB"
done
for ((i = 0; i < N; i++)); do [ -f "$D/$i" ] && cat "$D/$i" >> "$PART"; done
echo "size $(stat -c %s "$PART") (want $TOTAL)"
echo "$SHA  $PART" | sha256sum -c - && rm -rf "$D" && echo "DOWNLOAD OK"

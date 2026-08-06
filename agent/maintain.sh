#!/bin/bash
# On-demand link maintenance for stimviz.
#
# Runs the whole verify-and-repair pipeline in the right order, rebuilds
# index.html, and reports. Does NOT commit or push — review the diff first.
#
#   ./agent/maintain.sh              full pass (re-checks everything not cached)
#   ./agent/maintain.sh --fresh      discard caches and re-check from scratch
#
# Order matters and is not obvious:
#   1 hostprobe  learn how each host answers a path that does not exist. Every
#                later verdict is read in this context, so it comes first.
#   2 verify     per-URL verdicts (LIVE / DEAD / UNVERIFIABLE)
#   3 catchall   corpus-level: many URLs landing on one page nobody asked for.
#                Must follow verify, which is what records where links land.
#   4 retarget   match cards to real collections via institution indexes
#   5 reanchor   move whatever cannot be shown to work to somewhere that does
#   6 searchlink cards left on a generic page get a subject search instead
#   7 dedupe     after 4-6, same-title cards can share a destination and become
#                genuine duplicates. Applied to the data AND the page.
#   8 build      regenerate index.html
#
# A full pass takes roughly 20-40 minutes; most of it is step 2, which is
# deliberately serialised per host (hammering a host in parallel provokes
# rate-limiting that looks exactly like bot-blocking and corrupts verdicts).

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$1" = "--fresh" ]; then
  echo "==> discarding caches"
  rm -f agent/verify-cache.json agent/ancestor-cache.json
fi

# The hourly discovery job, if it is ever re-enabled, reads discoveries.json at
# the start of a run and writes it back at the end. A maintenance pass that
# overlaps one used to be silently reverted, so refuse to run alongside it.
if launchctl list 2>/dev/null | grep -q com.stimviz.discovery; then
  echo "!! com.stimviz.discovery is loaded; it can revert this pass mid-run."
  echo "   Stop it first:  launchctl bootout gui/$(id -u)/com.stimviz.discovery"
  exit 1
fi

echo "==> 1/8 host profiles (only probes hosts not already profiled)"
python3 agent/hostprobe.py

echo "==> 2/8 verify every URL"
python3 agent/verify.py

echo "==> 3/8 catch-all redirect detection"
python3 agent/catchall.py --min 3 --apply

echo "==> 4/8 match cards to real collections where an index exists"
python3 agent/retarget.py --apply --min 0.85

echo "==> 5/8 re-anchor anything that cannot be shown to work"
python3 agent/reanchor.py

echo "==> 6/8 subject-search fallback for cards left on a generic page"
python3 agent/searchlink.py --apply

echo "==> 7/8 drop cards duplicating another card's title AND destination"
python3 - <<'PY'
import json
p = "discoveries.json"
D = json.load(open(p))
seen, kept = set(), []
for c in D:
    key = (c.get("url", "").rstrip("/").lower(), c.get("name", "").strip().lower())
    if not key[0] or key in seen:
        continue
    seen.add(key)
    kept.append(c)
print(f"    {len(D)} -> {len(kept)} entries ({len(D)-len(kept)} duplicates dropped)")
json.dump(kept, open(p, "w"), indent=2, ensure_ascii=False)
PY

echo "==> 8/8 rebuild index.html"
python3 - <<'PY'
import json, sys
sys.path.insert(0, "agent")
from discover import build_html
build_html(json.load(open("discoveries.json")))
PY

echo "==> audit: any dead links left on the page?"
python3 agent/audit_page.py

cat <<'EOF'

Done. Nothing has been committed.

  git diff --stat
  git add discoveries.json index.html && git commit && git push origin main

Then confirm the deploy actually published — Pages builds here take ~4 minutes
and have failed silently before:

  gh api repos/swoonjet/stimviz/pages/builds --jq '.[0] | {status,commit,duration}'
EOF

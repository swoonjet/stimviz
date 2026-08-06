#!/usr/bin/env python3
"""
Find catch-all redirect targets across the whole corpus and condemn the links
that land on them.

Some sites answer an unknown path with a redirect to a generic page: HTTP 200,
real content, no "not found" anywhere. Checked one at a time such a link looks
healthy. Checked together the pattern is obvious — hundreds of different
requested URLs all landing on one page that none of them asked for.

That corpus-level view is the only way to see it, so it runs as its own pass
after verify.py.

Rule: if N or more distinct requested URLs redirect to the same final URL, and
that final URL is not itself what any of them asked for, the requested paths do
not exist. Their verdicts become DEAD (reason: catch_all_redirect), and the
final URL is recorded so re-anchoring never adopts it either.

Usage:  python3 agent/catchall.py [--min 3] [--apply]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from linkcheck import DEAD, LIVE, _norm_for_compare

PROJECT_DIR = Path(__file__).parent.parent
VERIFY_CACHE = PROJECT_DIR / "agent" / "verify-cache.json"
CATCHALL_FILE = PROJECT_DIR / "agent" / "catchall-targets.json"


def main():
    a = sys.argv
    min_hits = int(a[a.index("--min") + 1]) if "--min" in a else 3
    apply = "--apply" in a

    cache = json.loads(VERIFY_CACHE.read_text())

    # Curated roots are destinations we chose on purpose and confirmed. A
    # redirect from one of them (bl.uk/catalogues-and-collections -> /collection)
    # is a site rename, not a dead link, so never condemn them here.
    roots_file = PROJECT_DIR / "agent" / "domain-roots.json"
    curated = set()
    if roots_file.exists():
        spec = json.loads(roots_file.read_text())
        curated = {_norm_for_compare(u) for u in (spec.get("roots") or {}).values()}

    # Group requested URLs by where they actually landed.
    landed = defaultdict(set)
    for u, r in cache.items():
        fin = r.get("final")
        if r.get("verdict") == LIVE and fin and _norm_for_compare(fin) != _norm_for_compare(u):
            if _norm_for_compare(u) in curated:
                continue      # a curated root that was renamed, not a dead link
            landed[_norm_for_compare(fin)].add(u)

    requested = {_norm_for_compare(u) for u in cache}

    targets = {}
    for fin_norm, srcs in landed.items():
        if len(srcs) < min_hits:
            continue
        # If the destination is itself one of the catalogued URLs, a shared
        # landing page is plausible rather than a catch-all.
        if fin_norm in requested and len(srcs) < min_hits * 4:
            continue
        targets[fin_norm] = sorted(srcs)

    total = sum(len(v) for v in targets.values())
    print(f"{len(cache)} verdicts | {len(landed)} distinct redirect targets")
    print(f"catch-all targets (>= {min_hits} distinct sources): {len(targets)}")
    print(f"links landing on them: {total}\n")

    for fin, srcs in sorted(targets.items(), key=lambda kv: -len(kv[1]))[:20]:
        print(f"  {len(srcs):5}  {fin[:76]}")

    CATCHALL_FILE.write_text(json.dumps(
        {"min_hits": min_hits,
         "targets": {k: len(v) for k, v in sorted(targets.items(), key=lambda kv: -len(kv[1]))}},
        indent=1))
    print(f"\nwrote {CATCHALL_FILE}")

    if not apply:
        print("\n(dry run — pass --apply to rewrite verdicts)")
        return

    changed = 0
    for fin, srcs in targets.items():
        for u in srcs:
            rec = cache.get(u)
            if rec and rec.get("verdict") != DEAD:
                rec["verdict"] = DEAD
                rec["reason"] = "catch_all_redirect"
                rec["catch_all_target"] = fin
                changed += 1
    VERIFY_CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False))
    print(f"\nrewrote {changed} verdicts to DEAD (catch_all_redirect)")
    print(Counter(r["verdict"] for r in cache.values()).most_common())


if __name__ == "__main__":
    main()

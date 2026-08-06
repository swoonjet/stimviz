#!/usr/bin/env python3
"""
Audit the links actually embedded in index.html.

This is the gate that matters: it inspects what the page will serve, not what
the pipeline believed it was writing. It caught three separate bugs during the
2026-08-05/06 cleanup that the pipeline's own reporting had called success.

Checks:
  1 no card links to a URL with a DEAD verdict
  2 no card links to a known catch-all page
  3 every re-anchored card discloses what it opens (scope note)
  4 no two cards share both a title and a destination
  5 optional live sampling (--live N) — fetches N random links for real

Exit code is non-zero if any check fails, so it can gate a deploy.

Usage:  python3 agent/audit_page.py [--live 60]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "agent"))

INDEX = PROJECT_DIR / "index.html"
VERIFY = PROJECT_DIR / "agent" / "verify-cache.json"
ANCESTORS = PROJECT_DIR / "agent" / "ancestor-cache.json"
ROOTS = PROJECT_DIR / "agent" / "domain-roots.json"
PROFILE = PROJECT_DIR / "agent" / "host-profile.json"
CATCHALL = PROJECT_DIR / "agent" / "catchall-targets.json"


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def norm(u):
    u = (u or "").split("?")[0].split("#")[0]
    return re.sub(r"^www\.", "", re.sub(r"^https?://", "", u).rstrip("/").lower())


def main():
    argv = sys.argv
    live_n = int(argv[argv.index("--live") + 1]) if "--live" in argv else 0

    html = INDEX.read_text()
    m = re.search(r"const collections = (\[.*?\]);\n", html, re.S)
    if not m:
        print("FAIL: could not find the collections array in index.html")
        return 1
    cards = json.loads(m.group(1))

    verdicts = load_json(VERIFY, {})
    ancestors = load_json(ANCESTORS, {})
    roots = set((load_json(ROOTS, {}).get("roots") or {}).values())
    profiles = {r["host"]: r for r in load_json(PROFILE, [])}
    catchall = set(load_json(CATCHALL, {}).get("targets") or {})

    print(f"cards on page: {len(cards)}   unique destinations: "
          f"{len({c['url'] for c in cards})}")

    failures = []

    # 1 — dead links
    dead = Counter()
    buckets = Counter()
    for c in cards:
        u = c["url"]
        v = (verdicts.get(u) or {}).get("verdict") or (ancestors.get(u) or {}).get("verdict")
        if v == "DEAD":
            dead[u] += 1
        elif v == "LIVE":
            buckets["proven_live"] += 1
        elif c.get("scope") == "subject_search":
            buckets["verified_subject_search"] += 1
        elif u in roots:
            buckets["verified_curated_root"] += 1
        elif profiles.get(urlparse(u).netloc, {}).get("root_ok"):
            buckets["host_root_answers"] += 1
        else:
            buckets["unverifiable_other"] += 1
    for k, n in buckets.most_common():
        print(f"  {k:26} {n:6}")
    if dead:
        failures.append(f"{sum(dead.values())} cards link to a DEAD url")
        print(f"\nFAIL: {sum(dead.values())} cards link to a DEAD url")
        for u, n in dead.most_common(10):
            print(f"   {n:5}  {u[:70]}")
    else:
        print("  PASS  no card links to a DEAD url")

    # 2 — catch-all destinations
    onca = Counter(c["url"] for c in cards if norm(c["url"]) in catchall)
    if onca:
        # A host's own landing page is a legitimate destination even when dead
        # URLs also redirect there. nmbu.no/ redirects to nmbu.no/en/, so /en/
        # is that host's front door, not a wrong place to send a card.
        LOCALES = {"en", "en-us", "en-gb", "fr", "de", "es", "it", "nl", "no",
                   "nb", "nn", "sv", "da", "fi", "pt", "pl", "cs", "ja", "zh",
                   "ru", "index", "home"}

        def is_host_landing(u):
            p = urlparse(u)
            segs = [s for s in p.path.strip("/").split("/") if s]
            if not segs:
                return True
            prof = profiles.get(p.netloc) or {}
            if norm(prof.get("root_final") or "") == norm(u):
                return True
            # Many European institutions serve their front page at /en/.
            return len(segs) == 1 and segs[0].lower() in LOCALES

        bad = {u: n for u, n in onca.items() if not is_host_landing(u)}
        if bad:
            failures.append(f"{sum(bad.values())} cards link to a catch-all page")
            print(f"FAIL: {sum(bad.values())} cards link to a known catch-all page")
            for u, n in sorted(bad.items(), key=lambda kv: -kv[1])[:8]:
                print(f"   {n:5}  {u[:70]}")
        else:
            print(f"  PASS  catch-all pages used only as host roots "
                  f"({sum(onca.values())} cards)")
    else:
        print("  PASS  no card links to a known catch-all page")

    # 3 — disclosure on re-anchored cards
    fb = [c for c in cards if c.get("verified") == "fallback"]
    missing = [c for c in fb if not c.get("scope")]
    if missing:
        failures.append(f"{len(missing)} re-anchored cards lack a scope note")
        print(f"FAIL: {len(missing)} re-anchored cards do not disclose what they open")
    else:
        print(f"  PASS  all {len(fb)} re-anchored cards disclose their scope "
              f"{dict(Counter(c.get('scope') for c in fb))}")

    # 4 — duplicate cards
    dup = Counter((c["name"].strip().lower(), c["url"].rstrip("/").lower()) for c in cards)
    ndup = sum(n - 1 for n in dup.values() if n > 1)
    if ndup:
        failures.append(f"{ndup} duplicate cards (same title AND destination)")
        print(f"FAIL: {ndup} cards duplicate another card's title AND destination")
    else:
        print("  PASS  no card duplicates another's title and destination")

    # 5 — live sampling
    if live_n:
        import random
        from concurrent.futures import ThreadPoolExecutor
        from linkcheck import AuthoritativeIndex, HostProfiles, verdict_for
        prof, idx = HostProfiles(), AuthoritativeIndex()
        random.seed()
        sample = random.sample(cards, min(live_n, len(cards)))
        with ThreadPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(lambda c: (c["url"], c["name"][:40],
                                         verdict_for(c["url"], prof, idx)), sample))
        got = Counter(r[2]["verdict"] for r in res)
        print(f"\nlive sample of {len(sample)}: {dict(got)}")
        livedead = [(u, n, r) for u, n, r in res if r["verdict"] == "DEAD"]
        if livedead:
            failures.append(f"{len(livedead)} sampled links are DEAD live")
            print(f"FAIL: {len(livedead)} sampled links are DEAD when fetched")
            for u, n, r in livedead:
                print(f"   {r.get('reason','')[:26]:26} {u[:58]} ({n})")
        else:
            print("  PASS  no sampled link was dead when fetched")

    print()
    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

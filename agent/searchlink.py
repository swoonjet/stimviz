#!/usr/bin/env python3
"""
Send cards that would land on a generic browse page to a subject search instead.

Re-anchoring puts hundreds of cards on one collection root. The link works, but
"Posters and Broadsides from Imperial Russia" opening loc.gov/collections/ is not
related to what the card says. A search for the card's own subject terms on the
same institution lands on real, relevant material.

Only used where the institution's search is verified to work and to discriminate
(a nonsense query must return nothing — otherwise "results" prove nothing).

Every host pattern here was checked before use:
  loc.gov  /search/?q=<terms>   verified via ?fo=json — real terms return tens
                                of thousands of items, a nonsense term returns 0

Usage:
  python3 agent/searchlink.py --report
  python3 agent/searchlink.py --apply
"""

import json
import re
import subprocess
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"

# host -> (generic destination it would otherwise get, search URL template)
SEARCH_HOSTS = {
    "www.loc.gov": {
        "generic": ["https://www.loc.gov/collections/", "https://www.loc.gov/"],
        "template": "https://www.loc.gov/search/?q={q}",
        "probe": "https://www.loc.gov/search/?q={q}&fo=json&c=5",
    },
}

STOP = {
    "the", "of", "and", "a", "an", "at", "in", "on", "for", "from", "to", "with",
    "collection", "collections", "archive", "archives", "digital", "library",
    "libraries", "online", "items", "item", "database", "gallery", "project",
    "division", "department", "university", "center", "centre", "catalog",
    "catalogue", "congress", "loc", "various", "featuring", "including",
}


def query_terms(card, limit=5):
    """Build a short subject query from the card's own words.

    Short beats exact: the card titles are invented, so searching the full
    invented phrase often returns nothing, while its distinctive nouns return
    the real material the card describes.
    """
    name = card.get("name", "") or ""
    words, seen = [], set()
    for w in re.split(r"[^A-Za-z0-9]+", name):
        lw = w.lower()
        if len(w) > 2 and lw not in STOP and lw not in seen:
            seen.add(lw)
            words.append(w)
    if len(words) < 2:
        cat = card.get("category", "")
        if cat and cat.lower() not in seen:
            words.append(cat)
    return " ".join(words[:limit])


def probe_results(host, q):
    """Return the number of results the institution reports, or None."""
    spec = SEARCH_HOSTS[host]
    url = spec["probe"].format(q=urllib.parse.quote_plus(q))
    p = subprocess.run(["curl", "-s", "--max-time", "40", url],
                       capture_output=True, text=True)
    try:
        j = json.loads(p.stdout)
    except Exception:
        return None
    pg = j.get("pagination") or {}
    n = pg.get("of")
    if n is None:
        n = len(j.get("results") or [])
    return n


def main():
    a = sys.argv
    apply = "--apply" in a
    items = json.loads(DISCOVERIES.read_text())

    # Confirm the search endpoint still discriminates before relying on it.
    for host, spec in SEARCH_HOSTS.items():
        good = probe_results(host, "botanical illustration")
        junk = probe_results(host, "zzqfakenonsenseterm9471")
        print(f"{host} search check: real-term results={good}  nonsense={junk}")
        if not good or junk not in (0, None) and junk > 0:
            print(f"  REFUSING to use {host} search — it does not discriminate")
            SEARCH_HOSTS[host]["disabled"] = True

    targets = []
    for c in items:
        u = c.get("url", "") or ""
        host = urllib.parse.urlparse(u).netloc
        spec = SEARCH_HOSTS.get(host)
        if not spec or spec.get("disabled"):
            continue
        if u.rstrip("/") not in [g.rstrip("/") for g in spec["generic"]]:
            continue
        q = query_terms(c)
        if len(q) < 4:
            continue
        targets.append((c, host, q))

    print(f"\ncards sitting on a generic destination that could search instead: "
          f"{len(targets)}")
    for c, host, q in targets[:12]:
        print(f"  {c['name'][:52]:52} -> q={q!r}")

    # Sample-check that the generated queries actually return material.
    import random
    random.seed(5)
    sample = random.sample(targets, min(20, len(targets)))
    counts = []
    for c, host, q in sample:
        n = probe_results(host, q)
        counts.append((n if isinstance(n, int) else -1, q))
    hits = [n for n, _ in counts if n and n > 0]
    print(f"\nsampled {len(sample)} generated queries: "
          f"{len(hits)} returned results, "
          f"{sum(1 for n,_ in counts if n == 0)} returned nothing")
    for n, q in counts[:10]:
        print(f"   results={str(n):>8}  {q}")

    if not apply:
        print("\n(report only — pass --apply)")
        return

    n_applied = 0
    for c, host, q in targets:
        spec = SEARCH_HOSTS[host]
        c.setdefault("original_url", c.get("url", ""))
        c["url"] = spec["template"].format(q=urllib.parse.quote_plus(q))
        c["verified"] = "reanchored"
        c["link_scope"] = "subject_search"
        c["search_query"] = q
        n_applied += 1
    DISCOVERIES.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"\npointed {n_applied} cards at a subject search -> {DISCOVERIES}")


if __name__ == "__main__":
    main()

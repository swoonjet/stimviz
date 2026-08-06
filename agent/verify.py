#!/usr/bin/env python3
"""
Verify every URL in discoveries.json and cache the verdict.

Reads agent/host-profile.json (how each host behaves) and agent/indexes/*.json
(institution-published collection lists), then decides LIVE / DEAD /
UNVERIFIABLE for each URL. See agent/linkcheck.py for the decision rules.

Politeness matters here for correctness, not just courtesy: hammering one host
in parallel provokes rate-limiting, which looks exactly like bot-blocking and
would corrupt the verdicts. So URLs are grouped by host, each host is walked
serially with a delay, and parallelism happens across hosts.

Hosts that block every request, and hosts covered by an authoritative index,
need no requests at all — which removes most of the work.

Resumable: verdicts are cached in agent/verify-cache.json by URL.

Usage:  python3 agent/verify.py [--refresh] [--limit N]
"""

import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from linkcheck import (LIVE, DEAD, UNVERIFIABLE, AuthoritativeIndex,
                       HostProfiles, fetch_resilient, verdict_for)

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
CACHE = PROJECT_DIR / "agent" / "verify-cache.json"

HOST_WORKERS = 24
PER_HOST_DELAY = 0.7      # seconds between requests to the same host


def main():
    refresh = "--refresh" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    items = json.loads(DISCOVERIES.read_text())
    profiles = HostProfiles()
    indexes = AuthoritativeIndex()
    print(f"{len(items)} entries, {len(profiles.by_host)} host profiles, "
          f"{len(indexes.spaces)} authoritative index space(s)")

    cache = {}
    if CACHE.exists() and not refresh:
        try:
            cache = json.loads(CACHE.read_text())
        except Exception:
            cache = {}
    print(f"{len(cache)} cached verdicts")

    urls = []
    seen = set()
    for c in items:
        u = c.get("url")
        if u and u not in seen:
            seen.add(u)
            if u not in cache:
                urls.append(u)
    if limit:
        urls = urls[:limit]
    print(f"{len(seen)} unique URLs, {len(urls)} to check\n")
    if not urls:
        report(cache, items)
        return

    # Split into "needs a request" and "decidable without one".
    by_host = defaultdict(list)
    instant = []
    for u in urls:
        hcls = profiles.cls(u)
        v, _ = indexes.lookup(u)
        if v or hcls in ("BLOCKED", "UNREACHABLE"):
            instant.append(u)
        else:
            by_host[urlparse(u).netloc].append(u)

    print(f"decidable without a request: {len(instant)}")
    print(f"need a request:              {sum(len(v) for v in by_host.values())} "
          f"across {len(by_host)} hosts")
    biggest = sorted(by_host.items(), key=lambda kv: -len(kv[1]))[:6]
    for h, us in biggest:
        print(f"    {len(us):5}  {h}  (~{len(us)*(PER_HOST_DELAY+1.2)/60:.0f}m serial)")
    print()

    lock = threading.Lock()
    done = {"n": 0}
    total = len(instant) + sum(len(v) for v in by_host.values())
    started = time.time()

    def record(u, rec):
        with lock:
            cache[u] = rec
            done["n"] += 1
            n = done["n"]
            if n % 200 == 0 or n == total:
                el = time.time() - started
                rate = n / el if el else 0
                vc = Counter(r["verdict"] for r in cache.values())
                print(f"  {n:6}/{total}  {100*n/total:5.1f}%  {rate:5.1f}/s  "
                      f"eta {(total-n)/rate/60 if rate else 0:5.1f}m   "
                      f"LIVE={vc[LIVE]} DEAD={vc[DEAD]} UNVER={vc[UNVERIFIABLE]}",
                      flush=True)
            if n % 1000 == 0:
                CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False))

    # No-request verdicts first.
    for u in instant:
        record(u, verdict_for(u, profiles, indexes))

    # Then one worker per host, serial within the host.
    def walk_host(host_urls):
        for i, u in enumerate(host_urls):
            if i:
                time.sleep(PER_HOST_DELAY)
            try:
                rec = verdict_for(u, profiles, indexes)
            except Exception as e:
                rec = {"verdict": UNVERIFIABLE, "reason": f"error:{type(e).__name__}"}
            record(u, rec)

    with ThreadPoolExecutor(max_workers=HOST_WORKERS) as ex:
        list(ex.map(walk_host, by_host.values()))

    CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False))
    print(f"\ndone in {(time.time()-started)/60:.1f}m -> {CACHE}")
    report(cache, items)


def report(cache, items):
    print("\n" + "=" * 72)
    print("VERDICTS (by entry, not unique URL)")
    print("=" * 72)
    vc = Counter()
    rc = Counter()
    for c in items:
        r = cache.get(c.get("url", ""))
        if not r:
            vc["(unchecked)"] += 1
            continue
        vc[r["verdict"]] += 1
        rc[(r["verdict"], r.get("reason", "?"))] += 1
    tot = sum(vc.values())
    for k, v in vc.most_common():
        print(f"  {k:14} {v:6}  ({100*v/tot:5.1f}%)")
    print("\nreasons:")
    for (v, reason), n in rc.most_common(22):
        print(f"  {v:13} {reason:42} {n:6}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check URLs against the Wayback Machine CDX index.

Why this exists: about 45% of the hosts in this dataset answer every HTTP
request with a bot challenge, so a direct request cannot distinguish a real
collection page from an invented one. The Internet Archive's crawler is not
blocked by those hosts and it records the status code it saw, so its index can
answer the question when we cannot.

Validated behaviour (2026-08-05): known-real URLs on bot-walled hosts return
200-status captures; invented URLs return no captures at all.

A *recent* 200 capture (see --since) is the useful signal for a landing page:
it means the crawler successfully fetched that exact URL lately.

CDX is slow (8-20s per URL) and rate-limits, so this is for hundreds of URLs,
not tens of thousands. Results are cached in agent/cdx-cache.json.

Usage:
  python3 agent/cdxcheck.py --urls-file FILE [--since 2024] [--workers 4]
  python3 agent/cdxcheck.py --url URL [--since 2024]
"""

import json
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CACHE = PROJECT_DIR / "agent" / "cdx-cache.json"
ENDPOINT = "https://web.archive.org/cdx/search/cdx"

_lock = threading.Lock()


def query(url, since=None, timeout=90, retries=3):
    """Return {'captures': n, 'last': timestamp, 'ok': bool} for 200-status captures."""
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,statuscode",
        "filter": "statuscode:200",
        "collapse": "timestamp:6",
        "limit": "20",
    }
    if since:
        params["from"] = str(since)
    api = ENDPOINT + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries):
        p = subprocess.run(["curl", "-s", "--max-time", str(timeout), api],
                           capture_output=True, text=True)
        out = (p.stdout or "").strip()
        if out.startswith("[") or out == "":
            if out == "":
                return {"captures": 0, "last": None, "ok": False, "note": "empty"}
            try:
                rows = json.loads(out)
            except Exception:
                time.sleep(2 + attempt * 3)
                continue
            data = rows[1:] if rows and rows[0] and rows[0][0] == "timestamp" else rows
            if not data:
                return {"captures": 0, "last": None, "ok": False}
            stamps = sorted(r[0] for r in data if r and r[0])
            return {"captures": len(data), "last": stamps[-1] if stamps else None,
                    "ok": True}
        # 5xx / gateway timeout HTML — back off and retry
        time.sleep(3 + attempt * 5)
    return {"captures": 0, "last": None, "ok": False, "note": "cdx_unavailable"}


def check_many(urls, since=None, workers=4, cache_path=CACHE, label=""):
    cache = {}
    if Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except Exception:
            cache = {}

    key = lambda u: f"{u}|since={since or ''}"
    todo = [u for u in urls if key(u) not in cache]
    print(f"{label}{len(urls)} URLs, {len(urls)-len(todo)} cached, {len(todo)} to query")

    done = {"n": 0}
    t0 = time.time()

    def work(u):
        r = query(u, since=since)
        with _lock:
            cache[key(u)] = r
            done["n"] += 1
            if done["n"] % 10 == 0 or done["n"] == len(todo):
                el = time.time() - t0
                rate = done["n"] / el if el else 0
                print(f"  {done['n']:5}/{len(todo)}  {rate*60:.1f}/min  "
                      f"eta {(len(todo)-done['n'])/rate/60 if rate else 0:.1f}m", flush=True)
                Path(cache_path).write_text(json.dumps(cache, indent=0))

    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
        Path(cache_path).write_text(json.dumps(cache, indent=0))

    return {u: cache[key(u)] for u in urls if key(u) in cache}


def main():
    a = sys.argv
    since = a[a.index("--since") + 1] if "--since" in a else None
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 4

    if "--url" in a:
        u = a[a.index("--url") + 1]
        print(json.dumps(query(u, since=since), indent=1))
        return

    if "--urls-file" in a:
        f = Path(a[a.index("--urls-file") + 1])
        urls = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        res = check_many(urls, since=since, workers=workers)
        print()
        for u in urls:
            r = res.get(u, {})
            mark = "OK  " if r.get("captures") else "none"
            print(f"  {mark} caps={str(r.get('captures')):>3} last={str(r.get('last'))[:8]:>8}  {u}")
        return

    print(__doc__)


if __name__ == "__main__":
    main()

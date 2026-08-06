#!/usr/bin/env python3
"""
Profile every host in discoveries.json.

For each host, learn two things:

  1. How the host answers a path that definitely does not exist. This tells us
     whether HTTP status means anything on that host:
       404/410      HONEST   - status is trustworthy
       403/401/429  BLOCKED  - bot detection answers every path the same way
       200          SOFT404  - returns OK for anything, renders 404 in JS.
                               We keep the body text so real pages can later be
                               told apart from this host's 404 shell.
  2. Whether the host root is reachable, and where it lands. Re-anchoring needs
     a destination that actually works.

Writes agent/host-profile.json. Resumable: re-running only probes hosts that
are missing from the existing profile.

Usage:  python3 agent/hostprobe.py [--refresh]
"""

import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
PROFILE = PROJECT_DIR / "agent" / "host-profile.json"

FAKE_PATH = "zzq-stimviz-probe-not-a-real-path-9471"
UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
WORKERS = 20
TIMEOUT = 20

# Body/title markers that mean "a bot wall answered", not "this page exists".
CHALLENGE_MARKERS = (
    "just a moment", "attention required", "checking your browser",
    "request verification", "access denied", "enable javascript and cookies",
    "verifying connection", "site protection", "making sure you're not a bot",
    "making sure you&#39;re not a bot", "security checkpoint", "captcha",
    "azure waf", "cf-browser-verification", "ddos-guard",
)

TAG_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
STRIP_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def fetch(url, ua=None, timeout=TIMEOUT, max_bytes=200000):
    """GET via curl. Returns (code, body, effective_url). code is None on failure.

    curl's own UA gets through some WAFs that reject a spoofed browser UA
    (loc.gov is the clearest example), so the caller can choose.
    """
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout),
           "--max-filesize", str(max_bytes),
           "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "-H", "Accept-Language: en-US,en;q=0.9",
           "-w", "\n___META___%{http_code} %{url_effective}"]
    if ua:
        cmd += ["-A", ua]
    cmd.append(url)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return None, "", ""
    out = p.stdout or ""
    if "___META___" not in out:
        return None, "", ""
    body, meta = out.rsplit("\n___META___", 1)
    code_s, _, eff = meta.partition(" ")
    try:
        code = int(code_s)
    except ValueError:
        code = None
    return code, body, eff.strip()


def visible_text(body, limit=2500):
    """Strip scripts/styles/tags down to visible text, collapsed."""
    t = TAG_RE.sub(" ", body or "")
    t = STRIP_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()[:limit]


def title_of(body):
    m = TITLE_RE.search(body or "")
    return WS_RE.sub(" ", m.group(1)).strip()[:120] if m else ""


def looks_challenged(code, body, title):
    hay = (title + " " + (body or "")[:6000]).lower()
    if code in (403, 401, 429):
        return True
    return any(m in hay for m in CHALLENGE_MARKERS)


def probe_host(host):
    """Probe one host: a definitely-fake path, plus the root."""
    rec = {"host": host}

    # --- fake path: learn what "does not exist" looks like here ---
    code, body, eff = fetch(f"https://{host}/{FAKE_PATH}", ua=UA_BROWSER)
    if code is None:
        # retry once with curl's own UA before calling it unreachable
        code, body, eff = fetch(f"https://{host}/{FAKE_PATH}", ua=None)

    title = title_of(body)
    rec["fake_code"] = code
    rec["fake_title"] = title
    rec["fake_text"] = visible_text(body, 1800)
    rec["fake_len"] = len(body or "")

    # curl reports 000 when the request never completed (DNS failure, refused,
    # TLS failure) — that is an unreachable host, not an HTTP status.
    if code is None or code == 0:
        rec["class"] = "UNREACHABLE"
    elif looks_challenged(code, body, title):
        rec["class"] = "BLOCKED"
    elif code in (404, 410):
        rec["class"] = "HONEST"
    elif code == 200:
        rec["class"] = "SOFT404"
    elif 300 <= code < 400:
        rec["class"] = "SOFT404"
    elif code >= 500:
        rec["class"] = "SERVER_ERROR"
    else:
        rec["class"] = f"OTHER_{code}"

    # --- root: is there a working destination to re-anchor to? ---
    rcode, rbody, reff = fetch(f"https://{host}/", ua=UA_BROWSER)
    if rcode is None:
        rcode, rbody, reff = fetch(f"https://{host}/", ua=None)
    rtitle = title_of(rbody)
    rec["root_code"] = rcode
    rec["root_title"] = rtitle
    rec["root_final"] = reff
    rec["root_len"] = len(rbody or "")
    # A root is usable as a re-anchor target if it answers at all and is not a
    # hard error. A bot wall is fine here: a human's browser will render it.
    rec["root_ok"] = bool(
        rcode and rcode != 0
        and (rcode == 200 or looks_challenged(rcode, rbody, rtitle))
        and rcode not in (404, 410)
    )
    return rec


def main():
    refresh = "--refresh" in sys.argv
    items = json.loads(DISCOVERIES.read_text())

    counts = Counter()
    for c in items:
        h = urlparse(c.get("url", "")).netloc
        if h:
            counts[h] += 1

    existing = {}
    if PROFILE.exists() and not refresh:
        try:
            existing = {r["host"]: r for r in json.loads(PROFILE.read_text())}
        except Exception:
            existing = {}

    todo = [h for h in counts if h not in existing]
    print(f"{len(counts)} hosts across {len(items)} entries")
    print(f"{len(existing)} already profiled, probing {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    done = {"n": 0}
    lock = threading.Lock()
    started = time.time()
    out = dict(existing)

    def work(h):
        try:
            r = probe_host(h)
        except Exception as e:
            r = {"host": h, "class": "UNREACHABLE", "error": f"{type(e).__name__}: {e}",
                 "root_ok": False}
        r["entries"] = counts[h]
        with lock:
            out[h] = r
            done["n"] += 1
            n = done["n"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - started
                rate = n / el if el else 0
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n:5}/{len(todo)}  {100*n/len(todo):5.1f}%  "
                      f"{rate:.1f}/s  eta {eta/60:.1f}m", flush=True)
                PROFILE.write_text(json.dumps(sorted(out.values(),
                                                     key=lambda r: -r.get("entries", 0)),
                                              indent=1, ensure_ascii=False))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))

    rows = sorted(out.values(), key=lambda r: -r.get("entries", 0))
    PROFILE.write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    print(f"\ndone in {(time.time()-started)/60:.1f}m -> {PROFILE}")
    by_hosts = Counter(r["class"] for r in rows)
    by_entries = Counter()
    for r in rows:
        by_entries[r["class"]] += r.get("entries", 0)
    tot = sum(by_entries.values())
    print(f"\n{'class':14} {'hosts':>6} {'entries':>8}  share")
    for k, v in by_entries.most_common():
        print(f"{k:14} {by_hosts[k]:6} {v:8}  {100*v/tot:5.1f}%")
    roots_ok = sum(r.get("entries", 0) for r in rows if r.get("root_ok"))
    print(f"\nentries whose host root is a usable re-anchor target: "
          f"{roots_ok}/{tot} ({100*roots_ok/tot:.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Audit existing discoveries.json — content-aware URL validation.

Categorizes every URL as one of:
  live          — HTTP 200, body looks like a real collection page
  soft_404      — HTTP 200 but body says "page not found" / "404" in title
  hard_broken   — HTTP 4xx/5xx (not 403) or DNS failure or connection refused
  bot_blocked   — 403 / Cloudflare-challenge / can't determine
  timeout       — request timed out

For non-live URLs, computes an institution-root fallback (per-domain) and
re-tests it. Writes a JSON report to agent/audit-report.json. Read-only —
does not modify discoveries.json.
"""

import json
import re
import ssl
import sys
import time
import gzip
import io
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from collections import Counter

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
REPORT = PROJECT_DIR / "agent" / "audit-report.json"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
}
TIMEOUT = 10
WORKERS = 16

NOT_FOUND_PATTERNS = [
    r"\bpage not found\b",
    r"\b404 (?:error|not found|page)\b",
    r"\bpage you (?:are looking for|requested) (?:could not be|cannot be|was not) found\b",
    r"\bthis page (?:does not exist|cannot be found|isn't here)\b",
    r"\bno longer available\b",
    r"\bpage has moved\b",
    r"\bwe (?:can'?t|cannot|could not) find (?:that|the page)\b",
    r"\bsorry, (?:that page|this page|the page)\b",
    r"\bbroken link\b",
    r"\b(?:url|page) does not exist\b",
    r"\bnot found on this server\b",
    r"\boops!? (?:looks like|the page|something)\b",
    r"\bfile not found\b",
]
NF_RE = re.compile("|".join(NOT_FOUND_PATTERNS), re.IGNORECASE)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def fetch(url, timeout=TIMEOUT, max_redirects=5):
    """GET a URL, follow redirects manually, return (status, final_url, body_snippet)."""
    if not url or not url.startswith("http"):
        return ("invalid", url, "")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    current = url
    for _ in range(max_redirects):
        try:
            req = urllib.request.Request(current, headers=HEADERS, method="GET")
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                code = resp.status
                final = resp.geturl() or current
                # Read up to 8KB — enough for title + early body text
                raw = resp.read(8192)
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        raw = gzip.decompress(raw + b"\x00")
                    except Exception:
                        try:
                            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                        except Exception:
                            pass
                try:
                    body = raw.decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                return (code, final, body)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location")
                if loc:
                    current = loc if loc.startswith("http") else _join(current, loc)
                    continue
            return (e.code, current, "")
        except urllib.error.URLError as e:
            err = str(e)
            if "timed out" in err or "timeout" in err:
                return ("timeout", current, "")
            if "nodename" in err or "Name or service" in err or "getaddrinfo" in err:
                return ("dns", current, "")
            if "Connection refused" in err:
                return ("refused", current, "")
            if "SSL" in err or "certificate" in err:
                return ("ssl", current, "")
            return ("error", current, "")
        except Exception:
            return ("error", current, "")
    return ("redirect_loop", current, "")


def _join(base, loc):
    from urllib.parse import urljoin
    return urljoin(base, loc)


def classify(status, body):
    """Categorize a fetch result."""
    if status == "timeout":
        return "timeout"
    if status in ("dns", "refused", "invalid"):
        return "hard_broken"
    if status == 403 or status == 401:
        return "bot_blocked"
    if isinstance(status, int) and 400 <= status < 600:
        return "hard_broken"
    if status == 200:
        # Inspect body
        title_m = TITLE_RE.search(body)
        title = (title_m.group(1) if title_m else "").strip()[:200]
        # Soft-404 check on title (highest signal) and first 3KB of body
        if NF_RE.search(title):
            return "soft_404"
        snippet = re.sub(r"<[^>]+>", " ", body[:3000])
        if NF_RE.search(snippet):
            return "soft_404"
        # Cloudflare / bot challenge in body
        if "Just a moment" in title or "Attention Required" in title or "Cloudflare" in title:
            return "bot_blocked"
        if len(body.strip()) < 200:
            return "hard_broken"  # blank or near-blank
        return "live"
    return "error"


def get_title(body):
    m = TITLE_RE.search(body or "")
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1)).strip()
    return t[:200]


# Per-domain known good roots — for top domains we use a curated entry,
# for anything else we fall back to scheme://host (or scheme://host/collections).
DOMAIN_ROOTS = {
    "digitalcollections.nypl.org": "https://digitalcollections.nypl.org",
    "www.loc.gov":                 "https://www.loc.gov/collections/",
    "www.metmuseum.org":            "https://www.metmuseum.org/art/the-collection",
    "www.moma.org":                 "https://www.moma.org/collection",
    "collections.vam.ac.uk":        "https://collections.vam.ac.uk/",
    "americanhistory.si.edu":       "https://americanhistory.si.edu/collections",
    "quod.lib.umich.edu":           "https://quod.lib.umich.edu/",
    "ufdc.ufl.edu":                 "https://ufdc.ufl.edu/",
    "www.biodiversitylibrary.org":  "https://www.biodiversitylibrary.org/",
    "libraries.ucsd.edu":           "https://library.ucsd.edu/dc/",
    "digitalcollections.lib.washington.edu": "https://digitalcollections.lib.washington.edu/",
    "digital.library.ucla.edu":     "https://digital.library.ucla.edu/",
    "dp.la":                        "https://dp.la/browse-by-topic",
    "gallica.bnf.fr":               "https://gallica.bnf.fr/",
    "digital.lib.uiowa.edu":        "https://digital.lib.uiowa.edu/",
    "www.si.edu":                   "https://www.si.edu/collections",
    "www.britishmuseum.org":        "https://www.britishmuseum.org/collection",
    "www.rijksmuseum.nl":           "https://www.rijksmuseum.nl/en/collection",
    "www.getty.edu":                "https://www.getty.edu/art/collection/",
    "wellcomecollection.org":       "https://wellcomecollection.org/collections",
    "europeana.eu":                 "https://www.europeana.eu/en",
    "www.europeana.eu":             "https://www.europeana.eu/en",
}


def institution_root(url):
    try:
        p = urlparse(url)
    except Exception:
        return None
    if not p.netloc:
        return None
    if p.netloc in DOMAIN_ROOTS:
        return DOMAIN_ROOTS[p.netloc]
    # Default fallback: bare host
    return f"{p.scheme or 'https'}://{p.netloc}"


def main():
    items = json.loads(DISCOVERIES.read_text())
    items = [c for c in items if isinstance(c, dict) and c.get("url")]
    print(f"Auditing {len(items)} URLs with {WORKERS} workers...")
    print()

    results = [None] * len(items)
    fallback_cache = {}    # institution_root_url -> classification
    fallback_lock = threading.Lock()
    progress = {"done": 0, "live": 0, "soft": 0, "broken": 0, "blocked": 0, "timeout": 0}
    plock = threading.Lock()

    def task(i, item):
        url = item["url"]
        status, final_url, body = fetch(url)
        cls = classify(status, body)
        title = get_title(body) if cls in ("live", "soft_404") else ""

        rec = {
            "i": i,
            "url": url,
            "name": item.get("name", "")[:120],
            "institution": item.get("institution", "")[:80],
            "status": cls,
            "http": status,
            "final_url": final_url if final_url != url else None,
            "title": title,
        }

        if cls != "live":
            root = institution_root(url)
            if root and root.rstrip("/") != url.rstrip("/"):
                with fallback_lock:
                    cached = fallback_cache.get(root)
                if cached is None:
                    fb_status, fb_final, fb_body = fetch(root)
                    fb_cls = classify(fb_status, fb_body)
                    cached = {"url": root, "status": fb_cls, "http": fb_status}
                    with fallback_lock:
                        fallback_cache[root] = cached
                rec["fallback"] = cached

        with plock:
            progress["done"] += 1
            if cls == "live": progress["live"] += 1
            elif cls == "soft_404": progress["soft"] += 1
            elif cls == "bot_blocked": progress["blocked"] += 1
            elif cls == "timeout": progress["timeout"] += 1
            else: progress["broken"] += 1
            d = progress["done"]
            if d % 100 == 0 or d == len(items):
                pct = 100 * d / len(items)
                print(f"  {d:5}/{len(items)}  ({pct:5.1f}%)  "
                      f"live={progress['live']}  soft={progress['soft']}  "
                      f"broken={progress['broken']}  blocked={progress['blocked']}  "
                      f"timeout={progress['timeout']}", flush=True)

        return rec

    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(task, i, c): i for i, c in enumerate(items)}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["i"]] = rec

    elapsed = time.time() - started
    print()
    print(f"Done in {elapsed:.0f}s.")
    print()

    # Summary
    by_status = Counter(r["status"] for r in results if r)
    fallback_helps = sum(1 for r in results
                         if r and r["status"] != "live"
                         and r.get("fallback", {}).get("status") == "live")

    summary = {
        "audited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "by_status": dict(by_status),
        "fallback_can_recover": fallback_helps,
        "elapsed_seconds": round(elapsed, 1),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in by_status.most_common():
        print(f"  {k:14}  {v:5}  ({100*v/len(results):.1f}%)")
    print(f"  {'fallback_helps':14}  {fallback_helps:5}  (would resolve to a working institution root)")
    print()

    REPORT.write_text(json.dumps({
        "summary": summary,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"Report written to: {REPORT}")
    print()
    print("Top broken domains:")
    broken_domains = Counter(
        urlparse(r["url"]).netloc for r in results
        if r and r["status"] in ("hard_broken", "soft_404")
    )
    for d, n in broken_domains.most_common(15):
        print(f"  {n:4}  {d}")


if __name__ == "__main__":
    main()

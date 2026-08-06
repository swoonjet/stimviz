#!/usr/bin/env python3
"""
Re-anchor every link that cannot be shown to work.

Policy (chosen 2026-08-05): the site must not serve 404s. A link that is proven
LIVE keeps its exact URL. Everything else is moved to the nearest destination
that does work, preferring the most specific one available:

  1. the deepest ancestor of the original path that verifies LIVE
     (/a/b/c -> /a/b/ -> /a/) — keeps as much specificity as possible
  2. a curated collection root for that domain (agent/domain-roots.json)
  3. the host root, when the host answers at all
  4. no working destination -> drop the entry

Ancestor probing is what stops this from collapsing thousands of cards onto a
handful of bare domains. It only helps on hosts where HTTP status is
meaningful; on bot-walled hosts we fall straight through to step 2 or 3.

Every rewritten entry keeps its original URL in `original_url` and is marked
`link_scope: "collection_root"` so the page can be honest that the link goes to
the institution's collection rather than the specific sub-collection named.

Writes discoveries.json in place (backing up first) and prints a full report.

Usage:  python3 agent/reanchor.py [--dry-run] [--limit N]
"""

import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).parent))
from linkcheck import (LIVE, DEAD, UNVERIFIABLE, AuthoritativeIndex,
                       HostProfiles, verdict_for)

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
VERIFY_CACHE = PROJECT_DIR / "agent" / "verify-cache.json"
ANCESTOR_CACHE = PROJECT_DIR / "agent" / "ancestor-cache.json"
DOMAIN_ROOTS = PROJECT_DIR / "agent" / "domain-roots.json"
BACKUP = PROJECT_DIR / "discoveries.before-reanchor.json"
REPORT = PROJECT_DIR / "agent" / "reanchor-report.json"

WORKERS = 20
PER_HOST_DELAY = 0.7


def ancestors(url):
    """Yield ancestor URLs from most to least specific, excluding the URL itself
    and excluding the bare host (handled separately as the last resort)."""
    p = urlparse(url)
    segs = [s for s in p.path.strip("/").split("/") if s]
    for k in range(len(segs) - 1, 0, -1):
        yield urlunparse((p.scheme or "https", p.netloc, "/" + "/".join(segs[:k]) + "/",
                          "", "", ""))


def host_root_candidates(url, profiles):
    """Destinations to try for a host, most specific first.

    Where the root redirects to is usually the better landing page, but never
    append a slash to it: root_final can be a file
    (https://www.ualberta.ca/en/index.html), and index.html/ is a 404.
    """
    p = urlparse(url)
    prof = profiles.by_host.get(p.netloc)
    if not prof or not prof.get("root_ok"):
        return []
    out = []
    final = prof.get("root_final") or ""
    if final.startswith("http") and urlparse(final).netloc == p.netloc:
        out.append(final)
    bare = f"https://{p.netloc}/"
    if bare not in out:
        out.append(bare)
    return out


def main():
    dry = "--dry-run" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    items = json.loads(DISCOVERIES.read_text())
    # Back up before anything mutates `items`. The rewrite loop below edits these
    # dicts in place, so a backup written afterwards would capture the rewritten
    # URLs rather than the originals — which is exactly what happened once.
    if not dry and not BACKUP.exists():
        BACKUP.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        print(f"backed up originals -> {BACKUP}")

    profiles = HostProfiles()
    indexes = AuthoritativeIndex()
    verdicts = json.loads(VERIFY_CACHE.read_text()) if VERIFY_CACHE.exists() else {}
    roots = {}
    if DOMAIN_ROOTS.exists():
        _spec = json.loads(DOMAIN_ROOTS.read_text())
        # The file wraps the map in a "roots" key alongside provenance fields.
        roots = _spec.get("roots", _spec) if isinstance(_spec, dict) else {}
        roots = {k: v for k, v in roots.items() if not k.startswith("_")}
    acache = json.loads(ANCESTOR_CACHE.read_text()) if ANCESTOR_CACHE.exists() else {}

    print(f"{len(items)} entries | {len(verdicts)} verdicts | "
          f"{len(roots)} curated domain roots | {len(acache)} cached ancestors")

    needs_move = []
    for c in items:
        u = c.get("url", "")
        v = (verdicts.get(u) or {}).get("verdict")
        if v != LIVE:
            needs_move.append(c)
    if limit:
        needs_move = needs_move[:limit]
    print(f"{len(needs_move)} entries need a new destination\n")

    # ---- probe candidate ancestors, grouped by host and serialised per host ----
    wanted = defaultdict(set)
    for c in needs_move:
        u = c["url"]
        host = urlparse(u).netloc
        hcls = profiles.cls(u)
        # Ancestor probing only tells the truth where status is meaningful.
        if hcls in ("HONEST", "SOFT404"):
            for a in ancestors(u):
                if a not in acache and a not in verdicts:
                    wanted[host].add(a)

    todo = {h: sorted(s) for h, s in wanted.items() if s}
    total = sum(len(v) for v in todo.values())
    print(f"probing {total} candidate ancestor URLs across {len(todo)} hosts")
    if total:
        lock = threading.Lock()
        done = {"n": 0}
        t0 = time.time()

        def walk(host_urls):
            for i, a in enumerate(host_urls):
                if i:
                    time.sleep(PER_HOST_DELAY)
                try:
                    rec = verdict_for(a, profiles, indexes)
                except Exception as e:
                    rec = {"verdict": UNVERIFIABLE, "reason": f"error:{type(e).__name__}"}
                with lock:
                    acache[a] = rec
                    done["n"] += 1
                    if done["n"] % 200 == 0 or done["n"] == total:
                        el = time.time() - t0
                        r = done["n"] / el if el else 0
                        print(f"  {done['n']:6}/{total}  {r:5.1f}/s  "
                              f"eta {(total-done['n'])/r/60 if r else 0:5.1f}m", flush=True)
                    if done["n"] % 1000 == 0:
                        ANCESTOR_CACHE.write_text(json.dumps(acache, indent=0))

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(walk, todo.values()))
        ANCESTOR_CACHE.write_text(json.dumps(acache, indent=0, ensure_ascii=False))
        print(f"ancestor probing done in {(time.time()-t0)/60:.1f}m\n")

    def live(u):
        for src in (verdicts, acache):
            if (src.get(u) or {}).get("verdict") == LIVE:
                return True
        return False

    dest_checked = {}

    def not_dead(u):
        """True unless this destination is known or found to be DEAD.

        Destinations are few relative to entries, so checking each one once and
        caching is cheap — and it is the difference between publishing a working
        link and publishing a 404.
        """
        for src in (verdicts, acache, dest_checked):
            v = (src.get(u) or {}).get("verdict")
            if v:
                return v != DEAD
        try:
            rec = verdict_for(u, profiles, indexes)
        except Exception:
            rec = {"verdict": UNVERIFIABLE, "reason": "dest_check_failed"}
        dest_checked[u] = rec
        return rec["verdict"] != DEAD

    # ---------------------------- rewrite ----------------------------
    out = []
    actions = Counter()
    moved_to = Counter()
    report_rows = []

    for c in items:
        u = c.get("url", "")
        v = (verdicts.get(u) or {}).get("verdict")

        if v == LIVE:
            c.pop("link_scope", None)
            c["verified"] = True
            c["verified_reason"] = (verdicts.get(u) or {}).get("reason", "")
            out.append(c)
            actions["kept_live"] += 1
            continue

        # Candidate destinations, most specific first.
        cands = []
        hcls = profiles.cls(u)
        if hcls in ("HONEST", "SOFT404"):
            for a in ancestors(u):
                if live(a):
                    cands.append((a, "ancestor"))
                    break
        r = roots.get(urlparse(u).netloc)
        if r:
            cands.append((r, "curated_root"))
        for hr in host_root_candidates(u, profiles):
            cands.append((hr, "host_root"))

        # Never move a card onto a destination that itself verifies DEAD. This
        # is the guard that catches bad curated roots and file-path roots
        # instead of publishing them.
        target, how = None, None
        for cand, method in cands:
            if not_dead(cand):
                target, how = cand, method
                break

        if not target:
            actions["dropped_no_destination"] += 1
            report_rows.append({"url": u, "action": "dropped", "verdict": v,
                                "host_class": hcls, "name": c.get("name", "")[:80]})
            continue

        if target.rstrip("/") == u.rstrip("/"):
            # Already at the destination; nothing to rewrite.
            c["verified"] = "unknown" if v == UNVERIFIABLE else False
            out.append(c)
            actions["kept_at_destination"] += 1
            continue

        c["original_url"] = u
        c["url"] = target
        c["verified"] = "reanchored"
        c["reanchor_method"] = how
        c["reanchor_from_verdict"] = v
        # Only claim collection-root scope when we actually stepped up a level.
        c["link_scope"] = "collection_root" if how != "ancestor" else "parent_collection"
        out.append(c)
        actions[f"reanchored_{how}"] += 1
        moved_to[target] += 1
        report_rows.append({"url": u, "action": f"reanchor_{how}", "to": target,
                            "verdict": v, "host_class": hcls,
                            "name": c.get("name", "")[:80]})

    print("=" * 72)
    print("ACTIONS")
    print("=" * 72)
    tot = sum(actions.values())
    for k, n in actions.most_common():
        print(f"  {k:28} {n:6}  ({100*n/tot:5.1f}%)")
    print(f"  {'TOTAL':28} {tot:6}")
    print(f"\nentries before: {len(items)}   after: {len(out)}   "
          f"net {len(out)-len(items):+d}")
    print(f"distinct re-anchor destinations: {len(moved_to)}")
    print("\nmost-crowded destinations:")
    for t, n in moved_to.most_common(12):
        print(f"  {n:5}  {t}")

    REPORT.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "actions": dict(actions),
        "before": len(items), "after": len(out),
        "distinct_destinations": len(moved_to),
        "crowded": moved_to.most_common(60),
        "rows": report_rows,
    }, indent=1, ensure_ascii=False))
    print(f"\nreport -> {REPORT}")

    if dry:
        print("\n--dry-run: discoveries.json not modified")
        return

    DISCOVERIES.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {DISCOVERIES} ({len(out)} entries)")


if __name__ == "__main__":
    main()

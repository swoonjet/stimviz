#!/usr/bin/env python3
"""
Cleanup discoveries.json based on audit-report.json.

Actions:
  live          → keep as-is, add verified=True
  bot_blocked   → keep, verified="unknown"
  timeout       → keep, verified="unknown"
  hard_broken   → if working fallback exists, swap URL to fallback (preserve
                  original_url, set verified="via_fallback"). Else: drop.
  soft_404      → same as hard_broken
  error         → same as hard_broken

Then rebuilds index.html via discover.py:build_html.
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
REPORT      = PROJECT_DIR / "agent" / "audit-report.json"
BACKUP      = PROJECT_DIR / "discoveries.before-cleanup.json"

# Allow importing build_html from discover.py
sys.path.insert(0, str(PROJECT_DIR / "agent"))
from discover import build_html


def main():
    audit = json.loads(REPORT.read_text())
    items = json.loads(DISCOVERIES.read_text())
    results = audit["results"]

    # Map by index for fast lookup. Audit was indexed against items where i = position
    # in the filtered list (only dict items with a url). Build the same filter to align.
    aligned = [c for c in items if isinstance(c, dict) and c.get("url")]
    if len(aligned) != len(results):
        print(f"WARNING: aligned items ({len(aligned)}) != audit results ({len(results)})")
        print("Re-running classification by URL match instead of index alignment.")
        by_url = {r["url"]: r for r in results if r}
    else:
        by_url = None

    cleaned = []
    actions = Counter()

    for c in items:
        if not isinstance(c, dict):
            actions["dropped_not_dict"] += 1
            continue
        url = c.get("url", "")
        if not url:
            actions["dropped_no_url"] += 1
            continue

        # Find audit result for this entry
        if by_url is not None:
            r = by_url.get(url)
        else:
            i = aligned.index(c)
            r = results[i] if i < len(results) else None

        if r is None:
            # Wasn't audited (shouldn't happen). Keep with unknown status.
            c["verified"] = "unknown"
            cleaned.append(c)
            actions["kept_unaudited"] += 1
            continue

        status = r["status"]

        if status == "live":
            c["verified"] = True
            cleaned.append(c)
            actions["kept_live"] += 1
        elif status in ("bot_blocked", "timeout"):
            c["verified"] = "unknown"
            cleaned.append(c)
            actions[f"kept_{status}"] += 1
        elif status in ("hard_broken", "soft_404", "error"):
            fb = r.get("fallback")
            if fb and fb.get("status") == "live":
                c["original_url"] = url
                c["url"] = fb["url"]
                c["verified"] = "via_fallback"
                cleaned.append(c)
                actions["redirected_to_fallback"] += 1
            else:
                actions[f"dropped_{status}"] += 1
        else:
            # unknown classification — keep but mark
            c["verified"] = "unknown"
            cleaned.append(c)
            actions[f"kept_other_{status}"] += 1

    # Backup original
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        print(f"Backed up original to: {BACKUP}")
    else:
        print(f"(backup already exists at {BACKUP}, not overwriting)")

    # Write cleaned
    DISCOVERIES.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    print(f"Wrote cleaned: {DISCOVERIES} ({len(cleaned)} entries)")
    print()

    print("=" * 60)
    print("ACTIONS")
    print("=" * 60)
    total = sum(actions.values())
    for k, v in actions.most_common():
        pct = 100 * v / total if total else 0
        print(f"  {k:30}  {v:5}  ({pct:.1f}%)")
    print(f"  {'TOTAL processed':30}  {total:5}")
    print()
    print(f"Before: {len(items)} entries")
    print(f"After:  {len(cleaned)} entries")
    print(f"Net:    {len(cleaned) - len(items):+d}")
    print()

    # Rebuild index.html
    print("Rebuilding index.html...")
    build_html(cleaned)
    print()
    print("Done.")


if __name__ == "__main__":
    main()

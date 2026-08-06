#!/usr/bin/env python3
"""
Point cards at the real collection they were describing.

Re-anchoring guarantees a link that works, but a card named "LoC Broadsides"
landing on /collections/ is not *related* to what the card says. Where an
institution publishes its own collection list (agent/indexes/*.json) we can do
much better: match the card's own words against the real collection titles and
link to that collection.

This is the repair the invented URLs were reaching for. The dataset says
`historic-american-buildings-survey`; the real collection is
`historic-american-buildings-landscapes-and-engineering-records`, and the card
text names it clearly enough to match.

Matching is deliberately conservative — a confidently wrong link is worse than a
generic one, so anything below the score threshold is left for re-anchoring to
handle.

Usage:
  python3 agent/retarget.py --report            # show match quality, change nothing
  python3 agent/retarget.py --apply [--min 0.55]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from linkcheck import AuthoritativeIndex

# A match must also beat the runner-up by this much. Without it, cards whose
# words fit several real collections equally get assigned one at random.
MIN_MARGIN = 0.12

PROJECT_DIR = Path(__file__).parent.parent
DISCOVERIES = PROJECT_DIR / "discoveries.json"
REPORT = PROJECT_DIR / "agent" / "retarget-report.json"

# Words that carry no distinguishing signal in this corpus.
STOP = {
    "the", "of", "and", "a", "an", "at", "in", "on", "for", "from", "to", "with",
    "collection", "collections", "archive", "archives", "digital", "library",
    "libraries", "online", "items", "item", "database", "gallery", "project",
    "papers", "records", "series", "division", "department", "university",
    "museum", "institute", "center", "centre", "national", "american", "us",
    "congress", "loc", "images", "image", "photographs", "photograph", "prints",
}


def tokens(s):
    return [w for w in re.split(r"[^a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in STOP]


def score(card_tokens, card_text, title):
    """Similarity of a card to a real collection title, 0..1.

    Scored against the TITLE's distinctive words, and deliberately hostile to
    the failure mode that matters: a short title matching on one common word.
    "Daguerreotypes" is a single token, so a plain recall score gave 1.0 to any
    card mentioning daguerreotypes anywhere in its description — which put
    "American Vernacular Photography Project" on the Daguerreotypes collection.
    """
    tt = tokens(title)
    if not tt or not card_tokens:
        return 0.0
    ct = set(card_tokens)
    hits = sum(1 for w in tt if w in ct)
    if hits == 0:
        return 0.0

    # Verbatim title inside the card's own text is decisive — but only for a
    # multi-word title. "Daguerreotypes" is one word, and finding it in a
    # description does not mean the card IS that collection: it put "American
    # Vernacular Photography Project" on the Daguerreotypes collection.
    tl = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", title.lower())).strip()
    if len(tt) >= 2 and len(tl) > 12 and tl in card_text:
        return 1.0

    recall = hits / len(tt)
    precision = hits / max(1, len(ct))
    f = 2 * recall * precision / (recall + precision)
    s = 0.7 * recall + 0.3 * f

    # One matched word is a coincidence, not an identification.
    if hits < 2:
        s *= 0.45
    return round(min(1.0, s), 4)


def best_match(card, titles_by_url):
    """Return (best, best_score, margin_over_runner_up).

    The margin matters as much as the score: when two real collections score
    alike, the card's words do not identify either one, so guessing is wrong.
    """
    name = card.get("name", "")
    desc = card.get("description", "")
    # Weight the name: it is what the card claims to be.
    ct = tokens(name) * 2 + tokens(name + " " + desc)
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ",
                  (name + " " + desc).lower())).strip()
    ranked = []
    for url, title in titles_by_url.items():
        s = score(ct, text, title)
        if s > 0:
            ranked.append((s, url, title))
    if not ranked:
        return None, 0.0, 0.0
    ranked.sort(reverse=True)
    top = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else 0.0
    return (top[1], top[2]), top[0], round(top[0] - runner, 4)


def main():
    a = sys.argv
    apply = "--apply" in a
    min_score = float(a[a.index("--min") + 1]) if "--min" in a else 0.72

    items = json.loads(DISCOVERIES.read_text())
    indexes = AuthoritativeIndex()
    if not indexes.spaces:
        print("no authoritative indexes present — nothing to do")
        return

    results = []
    for host, prefix, urls, name in indexes.spaces:
        titles_by_url = urls
        # Candidates: cards on this host, in this URL space, that are not
        # already pointing at a real member of the index.
        cands = []
        for c in items:
            u = c.get("url", "") or ""
            ou = c.get("original_url", "") or ""
            if host not in u and host not in ou:
                continue
            if AuthoritativeIndex.norm(u) in titles_by_url:
                continue     # already on a real collection
            cands.append(c)
        print(f"\nindex '{name}' ({len(titles_by_url)} real collections): "
              f"{len(cands)} candidate cards")

        for c in cands:
            bm, bs, margin = best_match(c, titles_by_url)
            m, s = (bm, bs) if bm else (None, 0.0)
            results.append({
                "margin": margin,
                "name": c.get("name", "")[:90],
                "current_url": c.get("url", ""),
                "original_url": c.get("original_url", ""),
                "match_url": m[0] if m else None,
                "match_title": m[1] if m else None,
                "score": s,
                "_card": c,
            })

    scored = sorted(results, key=lambda r: -r["score"])
    buckets = Counter()
    for r in scored:
        s = r["score"]
        b = ">=0.80" if s >= .8 else ">=0.70" if s >= .7 else ">=0.60" if s >= .6 \
            else ">=0.55" if s >= .55 else ">=0.40" if s >= .4 else "<0.40"
        buckets[b] += 1
    print("\nmatch score distribution:")
    for k in [">=0.80", ">=0.70", ">=0.60", ">=0.55", ">=0.40", "<0.40"]:
        if buckets[k]:
            print(f"  {k:8} {buckets[k]:5}")

    ok = [r for r in scored if r["score"] >= min_score and r.get("margin",0) >= MIN_MARGIN]
    print(f"\nwould retarget at score>={min_score} AND margin>={MIN_MARGIN}: {len(ok)}")
    rejected_margin = [r for r in scored if r["score"] >= min_score and r.get("margin",0) < MIN_MARGIN]
    print(f"rejected for ambiguity (score high, margin low): {len(rejected_margin)}")
    print("\nrejected as ambiguous (examples):")
    for r in rejected_margin[:8]:
        print(f"  s={r['score']:.2f} m={r.get('margin',0):.2f}  {r['name'][:48]:48}")
        print(f"        -> {(r['match_title'] or '')[:66]}")
    print("\nstrongest matches:")
    for r in scored[:14]:
        print(f"  {r['score']:.2f}  {r['name'][:52]:52}")
        print(f"        -> {r['match_title'][:70]}")
    print("\nborderline (just above threshold):")
    for r in [x for x in scored if min_score <= x["score"] < min_score + 0.06][:10]:
        print(f"  {r['score']:.2f}  {r['name'][:52]:52}")
        print(f"        -> {r['match_title'][:70]}")
    print("\njust below threshold (left to re-anchoring):")
    for r in [x for x in scored if min_score - 0.06 <= x["score"] < min_score][:8]:
        print(f"  {r['score']:.2f}  {r['name'][:52]:52}")
        print(f"        -> {(r['match_title'] or '')[:70]}")

    REPORT.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "_card"} for r in scored],
        indent=1, ensure_ascii=False))
    print(f"\nreport -> {REPORT}")

    if not apply:
        print("\n(report only — pass --apply to rewrite discoveries.json)")
        return

    n = 0
    for r in scored:
        if r["score"] < min_score or not r["match_url"]:
            continue
        if r.get("margin", 0) < MIN_MARGIN:
            continue
        c = r["_card"]
        c.setdefault("original_url", c.get("url", ""))
        c["url"] = r["match_url"]
        c["verified"] = True             # a member of the institution's own index
        c["verified_reason"] = "authoritative_index_match"
        c["matched_collection"] = r["match_title"]
        c["match_score"] = r["score"]
        c.pop("link_scope", None)        # points at a real specific collection
        c.pop("reanchor_method", None)
        n += 1
    DISCOVERIES.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"\nretargeted {n} cards onto real collections -> {DISCOVERIES}")


if __name__ == "__main__":
    main()

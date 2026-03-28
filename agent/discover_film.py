#!/usr/bin/env python3
"""
Film & Moving Image Archive Discovery Agent
Focused discovery of 8mm, 16mm, Super 8, home movies, amateur film,
found footage, film preservation, and moving image archives.
"""

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

# Import shared utilities from the main discovery agent
sys.path.insert(0, str(Path(__file__).parent))
from discover import (
    ollama_generate, check_url, load_existing, get_existing_urls,
    is_duplicate, build_html, CATEGORIES,
    COLLECTIONS_FILE, BATCH2_FILE, DISCOVERIES_FILE, MODEL
)

FILM_DOMAINS = [
    "8mm film and 8mm home movie archives",
    "16mm film collections and 16mm educational films",
    "Super 8 film archives and Super 8 amateur footage",
    "home movies and amateur film collections",
    "found footage archives and orphan film collections",
    "film preservation and film restoration archives",
    "moving image archives and motion picture collections",
    "vintage film reels and celluloid collections",
    "experimental film and avant-garde cinema archives",
    "documentary film archives and nonfiction film collections",
    "newsreel archives and historical news film footage",
    "educational film archives and classroom film collections",
    "small gauge film archives and amateur cinematography",
    "regional film archives and local history moving images",
]

NUM_ROUNDS = 3
DOMAINS_PER_ROUND = 4


def search_film_archives(domain: str, existing_urls: set, batch_label: str) -> list[dict]:
    """Ask the LLM to discover film/moving-image digital archives."""
    prompt = f"""You are a research librarian specializing in film archives, moving image preservation, and small-gauge film history.

Find 5 REAL, CURRENTLY ACCESSIBLE digital archives or online collections related to: {domain}

Focus on:
- University and museum film/video collections available online
- National and regional moving image archives with digital access
- Film preservation organizations with online viewable collections
- Home movie and amateur film digitization projects
- Newsreel and documentary film libraries with streaming access
- Experimental and avant-garde film archives online
- Government and institutional film collections (e.g. National Archives, BFI, etc.)

REQUIREMENTS:
- Each must have a real, working URL to a browsable or streamable online collection
- Must be freely accessible (no paywall required for browsing)
- Prefer institutional sources: universities, archives, museums, national libraries
- Include a mix of well-known and lesser-known/obscure collections
- NO generic search engines, NO social media, NO stock footage sites
- NO Wikipedia, NO YouTube channels, NO Netflix/streaming services
- Focus on FILM specifically (not general photography or art)

For each archive, provide EXACTLY this JSON format (no other text):
```json
[
  {{
    "name": "Collection Name",
    "institution": "Institution Name",
    "url": "https://exact-url-to-collection",
    "description": "2-3 sentence description focusing on film formats (8mm, 16mm, Super 8, 35mm), number of items, subject matter, and era. Be specific about the moving image content.",
    "era": "YYYY-YYYY or descriptive range",
    "category": "film"
  }}
]
```

Return ONLY the JSON array. No preamble, no explanation."""

    print(f"  [{batch_label}] Searching: {domain}")
    response = ollama_generate(prompt, temperature=0.8)

    if not response:
        return []

    # Extract JSON
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        candidates = json.loads(response[start:end])
    except json.JSONDecodeError:
        print(f"  [{batch_label}] Failed to parse JSON response")
        return []

    # Validate and filter
    valid = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        url = c.get("url", "")
        name = c.get("name", "")
        if not url or not name:
            continue
        if is_duplicate(url, existing_urls, []):
            print(f"  [{batch_label}] Skip duplicate: {name}")
            continue
        # Force category to film for most results
        cat = c.get("category", "").lower()
        if cat not in CATEGORIES:
            cat = "film"
        # Override non-film categories since this is a film-focused search
        if cat != "film":
            cat = "film"
        c["category"] = cat
        valid.append(c)

    return valid


def validate_urls(candidates: list[dict]) -> list[dict]:
    """Check which URLs are actually reachable."""
    validated = []
    for c in candidates:
        url = c.get("url", "")
        print(f"  Checking: {c['name']} ... ", end="", flush=True)
        if check_url(url):
            print("OK")
            validated.append(c)
        else:
            print("FAILED")
    return validated


def run():
    print("=" * 60)
    print("  FILM & MOVING IMAGE ARCHIVE DISCOVERY")
    print(f"  Model: {MODEL} via Ollama")
    print(f"  Rounds: {NUM_ROUNDS}, Domains/round: {DOMAINS_PER_ROUND}")
    print(f"  Total domains: {len(FILM_DOMAINS)}")
    print("=" * 60)

    # Check Ollama
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("\n  [error] Ollama not running. Start with: ollama serve")
        sys.exit(1)

    # Load existing collections for dedup
    existing = load_existing()
    existing_urls = get_existing_urls(existing)
    print(f"\n  Loaded {len(existing)} existing collections")
    print(f"  Known URLs: {len(existing_urls)}")

    # Load current discoveries to append to
    all_discoveries = []
    if DISCOVERIES_FILE.exists():
        try:
            all_discoveries = json.loads(DISCOVERIES_FILE.read_text())
        except Exception:
            pass
    pre_count = len(all_discoveries)

    # Shuffle domains and run rounds
    domains_pool = FILM_DOMAINS[:]
    random.shuffle(domains_pool)

    for round_num in range(NUM_ROUNDS):
        print(f"\n{'—' * 60}")
        print(f"  ROUND {round_num + 1}/{NUM_ROUNDS}")
        print(f"{'—' * 60}")

        # Pick domains for this round (cycle through if needed)
        start_idx = round_num * DOMAINS_PER_ROUND
        domains_this_round = []
        for i in range(DOMAINS_PER_ROUND):
            idx = (start_idx + i) % len(domains_pool)
            domains_this_round.append(domains_pool[idx])

        round_candidates = []
        for i, domain in enumerate(domains_this_round):
            label = f"R{round_num+1}.{i+1}"
            results = search_film_archives(domain, existing_urls, label)
            round_candidates.extend(results)
            time.sleep(1)

        # Deduplicate within round
        seen_urls = set()
        unique = []
        for c in round_candidates:
            url = c["url"].rstrip("/").lower()
            if url not in seen_urls and url not in existing_urls:
                seen_urls.add(url)
                unique.append(c)

        print(f"\n  Round {round_num + 1}: {len(unique)} unique candidates")

        # Validate URLs
        if unique:
            print(f"  Validating URLs...")
            validated = validate_urls(unique)
            print(f"  Validated: {len(validated)}/{len(unique)}")

            for v in validated:
                existing_urls.add(v["url"].rstrip("/").lower())
                all_discoveries.append(v)

    # Save discoveries (appending to existing)
    DISCOVERIES_FILE.write_text(json.dumps(all_discoveries, indent=2, ensure_ascii=False))
    new_count = len(all_discoveries) - pre_count
    print(f"\n  New film discoveries this run: {new_count}")
    print(f"  Total discoveries in file: {len(all_discoveries)}")
    print(f"  Saved to: {DISCOVERIES_FILE}")

    # Rebuild index.html with all collections merged
    merged = []
    for f in [COLLECTIONS_FILE, BATCH2_FILE]:
        if f.exists():
            try:
                merged.extend(json.loads(f.read_text()))
            except Exception:
                pass
    merged.extend(all_discoveries)

    # Final dedup by URL
    seen = set()
    deduped = []
    for c in merged:
        url = c.get("url", "").rstrip("/").lower()
        if url and url not in seen:
            seen.add(url)
            deduped.append(c)

    print(f"  Total unique collections: {len(deduped)}")
    build_html(deduped)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(deduped)} collections in index.html")
    print(f"  {new_count} new film archives discovered this session")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Find representative thumbnails for digital archive collections that are missing them.
Uses Wikimedia Commons API to find relevant images based on collection name/category.
Falls back to Ollama LLM to generate better search queries if initial search fails.
Handles rate limiting with exponential backoff and saves progress incrementally.
"""

import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
COLLECTIONS_FILE = PROJECT_DIR / "collections.json"
BATCH2_FILE = PROJECT_DIR / "collections-batch2.json"
DISCOVERIES_FILE = PROJECT_DIR / "discoveries.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral-small"

# Rate limiting
API_DELAY = 2.0  # seconds between Wikimedia API calls
BACKOFF_BASE = 30  # seconds for first 429 backoff
MAX_BACKOFF = 120  # max backoff seconds

# Import build_html from discover.py
sys.path.insert(0, str(Path(__file__).parent))
from discover import build_html


def wikimedia_search(query: str, limit: int = 1, retries: int = 3) -> list[str]:
    """Search Wikimedia Commons for images matching query. Returns list of thumb URLs.
    Handles 429 rate limiting with exponential backoff."""
    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",  # File namespace
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": "400",
        "format": "json",
    })
    url = f"https://commons.wikimedia.org/w/api.php?{params}"

    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": "DigitalCollectionsBot/1.0 (thumbnail finder; polite)"
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(BACKOFF_BASE * (2 ** attempt), MAX_BACKOFF)
                print(f"    [rate-limited] Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
                continue
            print(f"    [warn] Wikimedia API HTTP {e.code}: {e}")
            return []
        except Exception as e:
            print(f"    [warn] Wikimedia API error: {e}")
            return []

        pages = data.get("query", {}).get("pages", {})
        urls = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 999)):
            for info in page.get("imageinfo", []):
                mime = info.get("mime", "")
                if "image" not in mime:
                    continue
                thumb = info.get("thumburl", "")
                if thumb:
                    urls.append(thumb)
        return urls

    print(f"    [warn] All retries exhausted for query: {query}")
    return []


def build_search_query(collection: dict) -> list[str]:
    """Build a list of search queries from collection metadata, most specific first."""
    name = collection.get("name", "")
    institution = collection.get("institution", "")
    category = collection.get("category", "")
    description = collection.get("description", "")
    era = collection.get("era", "")

    queries = []

    # Strip generic words from name
    clean_name = name
    for word in ["Digital Collections", "Digital Collection", "Collection", "Collections",
                 "Digital", "Archive", "Archives", "Online", "Database"]:
        clean_name = clean_name.replace(word, "")
    clean_name = " ".join(clean_name.split()).strip(" -:,")

    # Most specific: name + category
    if clean_name:
        queries.append(f"{clean_name} {category}")

    # Name + institution
    if institution and clean_name:
        queries.append(f"{clean_name} {institution}")

    # Just the name
    if clean_name:
        queries.append(clean_name)

    # Key descriptive terms from description
    if description:
        first_sent = description.split(".")[0]
        words = first_sent.split()[:8]
        if len(words) > 3:
            queries.append(" ".join(words))

    # Fallback: category + era
    if category and era:
        queries.append(f"vintage {category} {era.split('-')[0]}")
    elif category:
        queries.append(f"vintage {category} archive")

    return queries


def ollama_suggest_query(collection: dict) -> str:
    """Ask Ollama to suggest a good Wikimedia Commons search query."""
    prompt = f"""I need to find a representative image on Wikimedia Commons for this digital archive collection:

Name: {collection.get('name', '')}
Institution: {collection.get('institution', '')}
Category: {collection.get('category', '')}
Description: {collection.get('description', '')}
Era: {collection.get('era', '')}

Suggest ONE short search query (3-5 words) that would find a good representative image on Wikimedia Commons. The image should visually represent this type of collection.

Reply with ONLY the search query, nothing else."""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 50}
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip().strip('"').strip("'").split("\n")[0]
    except Exception as e:
        print(f"    [warn] Ollama error: {e}")
        return ""


def find_thumbnail(collection: dict) -> str:
    """Find a thumbnail URL for a collection. Returns URL or empty string."""
    name = collection.get("name", "Unknown")
    print(f"  Finding thumbnail for: {name}")

    # Try our generated queries first
    queries = build_search_query(collection)
    for query in queries:
        print(f"    Trying: '{query}'")
        urls = wikimedia_search(query, limit=1)
        if urls:
            print(f"    Found: {urls[0][:80]}...")
            return urls[0]
        time.sleep(API_DELAY)

    # Fall back to LLM-suggested query
    print(f"    Asking LLM for better query...")
    llm_query = ollama_suggest_query(collection)
    if llm_query:
        print(f"    LLM suggests: '{llm_query}'")
        urls = wikimedia_search(llm_query, limit=1)
        if urls:
            print(f"    Found: {urls[0][:80]}...")
            return urls[0]
        time.sleep(API_DELAY)

    # Last resort: just search the category
    category = collection.get("category", "archive")
    fallback = f"vintage {category} historical"
    print(f"    Last resort: '{fallback}'")
    urls = wikimedia_search(fallback, limit=1)
    if urls:
        print(f"    Found: {urls[0][:80]}...")
        return urls[0]

    print(f"    No thumbnail found")
    return ""


def process_file(filepath: Path, thumbnail_key: str) -> int:
    """Process a JSON file, finding thumbnails for entries missing them.
    Saves progress after every successful thumbnail find.
    Returns count of thumbnails added."""
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return 0

    data = json.loads(filepath.read_text())
    total = len(data)
    missing = sum(1 for e in data if not e.get(thumbnail_key, ""))
    print(f"  {total} entries, {missing} missing thumbnails")

    count = 0
    for i, entry in enumerate(data):
        current = entry.get(thumbnail_key, "")
        if current:
            continue  # Already has a thumbnail

        thumb_url = find_thumbnail(entry)
        if thumb_url:
            entry[thumbnail_key] = thumb_url
            count += 1
            # Save progress after each find
            filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            time.sleep(API_DELAY)

    print(f"\n  Saved {filepath.name}: {count} thumbnails added (of {missing} missing)")
    return count


def main():
    print("=" * 60)
    print("  THUMBNAIL FINDER")
    print("  Using Wikimedia Commons API + Ollama fallback")
    print(f"  API delay: {API_DELAY}s between requests")
    print("=" * 60)

    total_added = 0

    # Process discoveries.json (uses "thumbnail" key based on build_html logic)
    print(f"\n--- Processing discoveries.json ---")
    total_added += process_file(DISCOVERIES_FILE, "thumbnail")

    # Process collections.json (uses "thumbnail_url" key)
    print(f"\n--- Processing collections.json ---")
    total_added += process_file(COLLECTIONS_FILE, "thumbnail_url")

    # Process collections-batch2.json (uses "thumbnail" key)
    print(f"\n--- Processing collections-batch2.json ---")
    total_added += process_file(BATCH2_FILE, "thumbnail")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL THUMBNAILS ADDED: {total_added}")
    print(f"{'=' * 60}")

    # Rebuild HTML
    print("\n  Rebuilding index.html...")
    merged = []
    for f in [COLLECTIONS_FILE, BATCH2_FILE]:
        if f.exists():
            try:
                merged.extend(json.loads(f.read_text()))
            except Exception:
                pass
    if DISCOVERIES_FILE.exists():
        try:
            merged.extend(json.loads(DISCOVERIES_FILE.read_text()))
        except Exception:
            pass

    # Deduplicate by URL
    seen = set()
    deduped = []
    for c in merged:
        url = c.get("url", "").rstrip("/").lower()
        if url and url not in seen:
            seen.add(url)
            deduped.append(c)

    build_html(deduped)
    print(f"\n  Done! {len(deduped)} total collections in index.html")


if __name__ == "__main__":
    main()

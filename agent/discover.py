#!/usr/bin/env python3
"""
Digital Archive Discovery Agent
Uses local Ollama LLM to research and discover accessible digital archives
from institutions, libraries, universities, and science organisations.
Validates URLs, deduplicates against existing collections, and outputs
structured JSON for the Visual Collections Index.
"""

import json
import os
import re
import sys
import time
import gzip
import io
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from urllib.parse import urlparse

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral-small"
PROJECT_DIR = Path(__file__).parent.parent
COLLECTIONS_FILE = PROJECT_DIR / "collections.json"
BATCH2_FILE = PROJECT_DIR / "collections-batch2.json"
DISCOVERIES_FILE = PROJECT_DIR / "discoveries.json"
INDEX_FILE = PROJECT_DIR / "index.html"

CATEGORIES = [
    "illustration", "poster", "design", "ephemera", "scientific",
    "map", "pattern", "fashion", "photo", "folk", "animation",
    "architecture", "typography", "audio", "film", "nature"
]

SEARCH_DOMAINS = [
    "art", "design", "audio and sound", "architecture and buildings",
    "typography and letterforms", "illustration and drawing",
    "science and scientific illustration", "natural history specimens and plates",
    "nature and botanical illustration", "ephemera and printed matter",
    "photography and daguerreotypes", "film and moving image",
    "graphic design and visual communication", "posters and broadsides",
    "maps and cartography", "patterns and textiles",
    "folk art and vernacular design", "animation and motion graphics",
    "fashion plates and costume", "music and audio archives",
    "zines and independent publishing", "book arts and printing",
    "industrial design and objects", "landscape and urban photography",
    "scientific instruments and diagrams", "ethnographic collections",
    "space and astronomy imagery", "marine biology illustrations",
    "ornithological plates and bird art", "entomology and insect illustration",
    "geological surveys and maps", "medical and anatomical illustration",
]


def ollama_generate(prompt: str, temperature: float = 0.7) -> str:
    """Call local Ollama instance."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4096}
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except Exception as e:
        print(f"  [error] Ollama call failed: {e}")
        return ""


_NF_PATTERNS = [
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
    r"\bfile not found\b",
]
_NF_RE = re.compile("|".join(_NF_PATTERNS), re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def is_acceptable_url_shape(url: str) -> bool:
    """Pre-filter URLs based on shape — reject query strings, deep paths, very long.
    These are the LLM-hallucination signature patterns from the audit."""
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return False
    if len(url) > 110:
        return False
    if "?" in url:
        return False
    if "#" in url and url.split("#", 1)[1].strip():
        # Hash routes occasionally legit; allow but only if base is short
        if len(url) > 90:
            return False
    try:
        p = urlparse(url)
    except Exception:
        return False
    if not p.netloc:
        return False
    depth = len([s for s in p.path.strip("/").split("/") if s])
    if depth > 4:
        return False
    return True


def check_url(url: str, timeout: int = 10) -> bool:
    """Validate a candidate URL. Admits it only on positive evidence that the
    page exists.

    This used to return True for 401/403 on the theory that a bot wall meant
    "the institution exists". It does mean that — but it says nothing about the
    path, and roughly 45% of the hosts here wall every request. So every
    invented URL on those hosts was admitted. A 2026-08-05 audit against the
    Library of Congress's own collection index found 96% of the loc.gov links
    admitted this way pointed at slugs that never existed.

    Now: a URL is admitted only if it verifies LIVE (see agent/linkcheck.py),
    or, when HTTP cannot decide, if the Internet Archive holds a real capture of
    that exact URL. Unverifiable is not the same as valid.
    """
    if not is_acceptable_url_shape(url):
        return False

    try:
        from linkcheck import (LIVE, UNVERIFIABLE, AuthoritativeIndex,
                               HostProfiles, verdict_for)
    except ImportError:
        print(" [linkcheck unavailable — refusing to admit unverified URL] ", end="")
        return False

    global _PROFILES, _INDEXES
    if _PROFILES is None:
        _PROFILES = HostProfiles()
        _INDEXES = AuthoritativeIndex()

    try:
        rec = verdict_for(url, _PROFILES, _INDEXES)
    except Exception:
        return False

    if rec["verdict"] == LIVE:
        return True
    if rec["verdict"] != UNVERIFIABLE:
        return False

    # HTTP could not decide. Ask the Internet Archive whether this exact URL was
    # ever really fetched. Invented URLs have no captures.
    if not ADMIT_VIA_WAYBACK:
        return False
    try:
        from cdxcheck import query as cdx_query
        r = cdx_query(url, timeout=45, retries=2)
    except Exception:
        return False
    if r.get("captures", 0) >= WAYBACK_MIN_CAPTURES:
        print(f" [wayback:{r['captures']} caps] ", end="")
        return True
    return False


# Lazily-loaded host behaviour profile + institution indexes (see check_url).
_PROFILES = None
_INDEXES = None

# Snapshot of discoveries.json as a run began; used to merge rather than clobber.
_baseline_discoveries = []

# When HTTP cannot decide, fall back to Internet Archive evidence.
ADMIT_VIA_WAYBACK = True
WAYBACK_MIN_CAPTURES = 1


def load_existing() -> list[dict]:
    """Load all existing collections to avoid duplicates."""
    existing = []
    for f in [COLLECTIONS_FILE, BATCH2_FILE, DISCOVERIES_FILE]:
        if f.exists():
            try:
                existing.extend(json.loads(f.read_text()))
            except Exception:
                pass
    return existing


def get_existing_urls(existing: list[dict]) -> set[str]:
    """Extract all known URLs for dedup."""
    urls = set()
    for c in existing:
        url = c.get("url", "").rstrip("/").lower()
        if url:
            urls.add(url)
    return urls


def is_duplicate(url: str, existing_urls: set[str], existing: list[dict]) -> bool:
    """Check if this exact URL is already known. Allows multiple collections
    from the same institution (e.g. LOC has many distinct collections)."""
    normalized = url.rstrip("/").lower()
    return normalized in existing_urls


def search_archives(domain: str, existing_urls: set[str], batch_num: int) -> list[dict]:
    """Ask the LLM to discover digital archives for a specific domain."""
    prompt = f"""You are a research librarian specializing in digital archives and open-access collections.

Find 5 REAL, CURRENTLY ACCESSIBLE digital archives or collections related to: {domain}

Focus on:
- University and college digital collections
- National and international library digitization projects
- Museum online collections with downloadable/browsable assets
- Scientific institution image databases
- Government archives and cultural heritage portals
- Independent curated digital archives

REQUIREMENTS:
- Each must have a real, working URL to a browsable online collection
- Must be freely accessible (no paywall required for browsing)
- Prefer institutions: universities, museums, national libraries, science orgs
- Include a mix of well-known and lesser-known/obscure collections
- NO generic search engines, NO social media, NO stock photo sites
- NO Wikipedia, NO Google Arts & Culture (too generic)

URL RULES — BE STRICT (most failures come from violating these):
- The URL MUST NOT contain a query string (no "?" anywhere)
- The URL MUST NOT exceed 110 characters
- The URL path MUST be 4 levels or fewer (e.g. /collections/posters is fine; /art/collection/search/topic/sub/12345 is too deep)
- POINT TO A BROWSE OR INDEX PAGE, never a specific item, search-result page, or API endpoint
- DO NOT include URLs that look like /search, /api/, /iiif/, .json, /item/123456
- If you do not know an exact, short URL, give the institution's main digital-collections page instead of guessing a deep path
- Better one short root URL that works than a long specific URL that 404s

For each archive, provide EXACTLY this JSON format (no other text):
```json
[
  {{
    "name": "Collection Name",
    "institution": "Institution Name",
    "url": "https://exact-url-to-collection",
    "description": "2-3 sentence description of what's in the collection, how many items, what formats, what era. Be specific.",
    "era": "YYYY-YYYY or descriptive range",
    "category": "one of: illustration, poster, design, ephemera, scientific, map, pattern, fashion, photo, folk, animation, architecture, typography, audio, film, nature"
  }}
]
```

Return ONLY the JSON array. No preamble, no explanation."""

    print(f"  [{batch_num}] Searching: {domain}")
    response = ollama_generate(prompt, temperature=0.8)

    if not response:
        return []

    # Extract JSON from response
    try:
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        json_str = response[start:end]
        candidates = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  [{batch_num}] Failed to parse JSON response")
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
        if not is_acceptable_url_shape(url):
            print(f"  [{batch_num}] Skip bad-shape URL: {url[:80]}")
            continue
        if is_duplicate(url, existing_urls, []):
            print(f"  [{batch_num}] Skip duplicate: {name}")
            continue
        # Validate category
        cat = c.get("category", "").lower()
        if cat not in CATEGORIES:
            cat = "design"  # default
        c["category"] = cat
        valid.append(c)

    return valid


def validate_urls(candidates: list[dict]) -> list[dict]:
    """Check which URLs are actually accessible."""
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


def build_html(all_collections: list[dict]):
    """Rebuild the index.html with all collections."""
    # Sort: confirmed-live first, re-anchored next, unconfirmed last; then
    # category, name.
    def _sort_key(c):
        v = c.get("verified")
        if v is True:
            rank = 0
        elif v in ("via_fallback", "reanchored"):
            rank = 1
        else:
            rank = 2
        return (rank, c.get("category", ""), c.get("name", ""))
    all_collections.sort(key=_sort_key)

    # Count stats
    total = len(all_collections)
    edu_count = sum(1 for c in all_collections if ".edu" in c.get("url", ""))
    intl_count = sum(1 for c in all_collections
                     if not any(tld in c.get("url", "")
                                for tld in [".gov", ".edu", ".com", ".org", ".io", ".net"])
                     or any(tld in c.get("url", "")
                            for tld in [".uk", ".nl", ".de", ".fr", ".jp", ".ch", ".au", ".it", ".se", ".dk", ".no", ".fi", ".be", ".at", ".nz"]))
    loc_count = sum(1 for c in all_collections if "loc.gov" in c.get("url", ""))
    verified_count = sum(1 for c in all_collections if c.get("verified") is True)
    categories = sorted(set(c.get("category", "design") for c in all_collections))
    cat_count = len(categories)

    # Build JS collections array
    js_entries = []
    for c in all_collections:
        v = c.get("verified")
        if v is True:
            verified = "live"
        elif v in ("via_fallback", "reanchored"):
            verified = "fallback"
        else:
            verified = "unknown"
        entry = {
            "name": c.get("name", ""),
            "institution": c.get("institution", ""),
            "url": c.get("url", ""),
            "description": c.get("description", ""),
            "era": c.get("era", ""),
            "thumbnail": c.get("thumbnail", c.get("thumbnail_url", "")),
            "category": c.get("category", "design"),
            "verified": verified,
        }
        # A re-anchored card does not link to the sub-collection it names, so
        # say which level it actually opens.
        if verified == "fallback":
            entry["scope"] = c.get("link_scope", "collection_root")
        js_entries.append(entry)

    collections_js = json.dumps(js_entries, indent=2, ensure_ascii=False)

    # Generate filter buttons
    all_cats = ["all"] + categories
    cat_labels = {
        "illustration": "Illustration", "poster": "Poster", "design": "Design",
        "ephemera": "Ephemera", "scientific": "Scientific", "map": "Map",
        "pattern": "Pattern", "fashion": "Fashion", "photo": "Photo",
        "folk": "Folk Art", "animation": "Animation", "architecture": "Architecture",
        "typography": "Typography", "audio": "Audio", "film": "Film",
        "nature": "Nature", "all": "All",
    }
    filter_buttons = "\n  ".join(
        f'<button class="filter-btn{" active" if cat == "all" else ""}" data-filter="{cat}">{cat_labels.get(cat, cat.title())}</button>'
        for cat in all_cats
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Collections Index — {total} Digital Archives</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0A0A0A;
    color: #E8E4DE;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}

  header {{
    padding: 60px 40px 40px;
    border-bottom: 1px solid #222;
  }}

  header h1 {{
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -1px;
    color: #fff;
    line-height: 1.1;
  }}

  header p {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #888;
    margin-top: 12px;
    max-width: 700px;
  }}

  .filters {{
    padding: 20px 40px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    border-bottom: 1px solid #1a1a1a;
    position: sticky;
    top: 0;
    background: #0A0A0A;
    z-index: 10;
  }}

  .filter-btn {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 6px 14px;
    border: 1px solid #333;
    background: transparent;
    color: #888;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  .filter-btn:hover {{ border-color: #C9A962; color: #C9A962; }}
  .filter-btn.active {{ border-color: #C9A962; color: #0A0A0A; background: #C9A962; }}

  .search-bar {{
    padding: 16px 40px;
    border-bottom: 1px solid #1a1a1a;
  }}

  .search-bar input {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 16px;
    border: 1px solid #333;
    background: #111;
    color: #E8E4DE;
    border-radius: 6px;
    width: 100%;
    max-width: 400px;
    outline: none;
    transition: border-color 0.2s;
  }}

  .search-bar input:focus {{ border-color: #C9A962; }}
  .search-bar input::placeholder {{ color: #555; }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1px;
    background: #1a1a1a;
    margin: 0;
  }}

  /* The whole card is the link: 14,000 entries are scanned, not read, so the
     click target is the card and there is no separate call-to-action row. */
  a.card {{
    background: #0f0f0f;
    padding: 0;
    display: flex;
    flex-direction: column;
    transition: background 0.15s;
    text-decoration: none;
    color: inherit;
  }}

  a.card:hover {{ background: #171717; }}
  a.card:hover .card-title {{ color: #C9A962; }}
  a.card:focus-visible {{ outline: 2px solid #C9A962; outline-offset: -2px; }}

  .card-body {{ padding: 13px 16px 14px; flex: 1; display: flex; flex-direction: column; }}

  .card-meta {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
  }}

  .card-category {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #C9A962;
  }}

  .card-era {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #4a4a4a;
    margin-left: auto;
  }}

  .card-title {{
    font-size: 14px;
    font-weight: 600;
    color: #fff;
    line-height: 1.3;
    margin-bottom: 3px;
    transition: color 0.15s;
  }}

  .card-institution {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6a6a6a;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  /* Two lines is enough to judge relevance while scanning; the rest is noise. */
  .card-desc {{
    font-size: 11.5px;
    color: #8c8c8c;
    line-height: 1.5;
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}

  .stats-bar {{
    display: flex;
    gap: 32px;
    padding: 20px 40px;
    border-bottom: 1px solid #1a1a1a;
    font-family: 'IBM Plex Mono', monospace;
    flex-wrap: wrap;
  }}

  .stat {{ display: flex; align-items: baseline; gap: 6px; }}
  .stat-num {{ font-size: 20px; font-weight: 600; color: #C9A962; }}
  .stat-label {{ font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }}

  footer {{
    padding: 40px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #333;
    border-top: 1px solid #1a1a1a;
  }}

  .new-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #0A0A0A;
    background: #6BCB77;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  /* Verification states */
  .v-dot {{
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: 1px;
    flex: none;
  }}
  .v-dot.live     {{ background: #6BCB77; }}
  .v-dot.fallback {{ background: #C9A962; }}
  .v-dot.unknown  {{ background: #555; }}

  /* Scope disclosure. Reads as one short line so it can be skimmed past, but a
     card must never silently imply it opens the sub-collection it names. */
  .v-note {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #6f6146;
    margin-top: 7px;
    line-height: 1.4;
  }}
  .v-note.unknown {{ color: #5a5a5a; }}

  a.card.unknown  {{ opacity: 0.8; }}
  a.card.unknown:hover {{ opacity: 1; }}

  .toggle-row {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .toggle-row label {{
    cursor: pointer;
    user-select: none;
    padding: 6px 12px;
    border: 1px solid #333;
    border-radius: 20px;
    transition: all 0.2s;
  }}
  .toggle-row input {{ display: none; }}
  .toggle-row input:checked + label {{ border-color: #6BCB77; color: #6BCB77; }}
</style>
</head>
<body>

<header>
  <h1>Visual Collections Index</h1>
  <p>{total} digital archives of illustration, photography, graphic design, typography, audio, film, science, natural history, architecture, and visual art from institutions worldwide.</p>
</header>

<div class="stats-bar">
  <div class="stat"><span class="stat-num">{total}</span><span class="stat-label">Collections</span></div>
  <div class="stat"><span class="stat-num">{verified_count}</span><span class="stat-label">Verified</span></div>
  <div class="stat"><span class="stat-num">{edu_count}</span><span class="stat-label">.edu sources</span></div>
  <div class="stat"><span class="stat-num">{intl_count}</span><span class="stat-label">International</span></div>
  <div class="stat"><span class="stat-num">{loc_count}</span><span class="stat-label">Library of Congress</span></div>
  <div class="stat"><span class="stat-num">{cat_count}</span><span class="stat-label">Categories</span></div>
</div>

<div class="search-bar">
  <input type="text" id="search" placeholder="Search collections, institutions, descriptions...">
</div>

<div class="filters">
  {filter_buttons}
  <div class="toggle-row">
    <input type="checkbox" id="verified-only">
    <label for="verified-only">verified only</label>
  </div>
</div>

<div class="grid" id="grid"></div>

<footer>Agent-curated collection &mdash; Last updated {time.strftime("%Y-%m-%d")} &mdash; All collections are publicly accessible.</footer>

<script>
const collections = {collections_js};

const grid = document.getElementById('grid');
const searchInput = document.getElementById('search');
let currentFilter = 'all';
let searchQuery = '';

const verifiedOnlyEl = document.getElementById('verified-only');
let verifiedOnly = false;

const linkLabel = {{
  live:     'Browse collection',
  fallback: 'Open institution',
  unknown:  'Open (unverified)'
}};
const noteText = {{
  fallback: 'original deep link unavailable — opens institution\\'s main collections page',
  unknown:  'unverified — could not confirm from outside the institution'
}};
// A re-anchored card links one level up from what it names. Say which level.
const scopeNote = {{
  parent_collection: 'original deep link unavailable — opens the parent collection',
  collection_root:   'original deep link unavailable — opens institution\\'s main collections page',
  subject_search:    'opens a search of this institution for this subject'
}};

function renderCards() {{
  let filtered = currentFilter === 'all' ? collections : collections.filter(c => c.category === currentFilter);
  if (verifiedOnly) {{
    filtered = filtered.filter(c => c.verified === 'live');
  }}
  if (searchQuery) {{
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(c =>
      c.name.toLowerCase().includes(q) ||
      c.institution.toLowerCase().includes(q) ||
      c.description.toLowerCase().includes(q) ||
      c.category.toLowerCase().includes(q)
    );
  }}
  grid.innerHTML = filtered.map(c => {{
    const v = c.verified || 'unknown';
    const noteMsg = (v === 'fallback' && scopeNote[c.scope]) || noteText[v];
    const note = noteMsg ? `<div class="v-note ${{v}}">${{noteMsg}}</div>` : '';
    return `
    <a class="card ${{v}}" data-category="${{c.category}}"
       href="${{c.url}}" target="_blank" rel="noopener"
       title="${{linkLabel[v] || linkLabel.unknown}} — ${{c.url}}">
      <div class="card-body">
        <div class="card-meta">
          <span class="card-category">${{c.category}}</span>
          <span class="card-era">${{c.era}}</span>
        </div>
        <h2 class="card-title"><span class="v-dot ${{v}}" title="${{v}}"></span>${{c.name}}</h2>
        <p class="card-institution">${{c.institution}}</p>
        <p class="card-desc">${{c.description}}</p>
        ${{note}}
      </div>
    </a>
  `;}}).join('');
}}

document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    renderCards();
  }});
}});

searchInput.addEventListener('input', (e) => {{
  searchQuery = e.target.value;
  renderCards();
}});

verifiedOnlyEl.addEventListener('change', (e) => {{
  verifiedOnly = e.target.checked;
  renderCards();
}});

renderCards();
</script>
</body>
</html>'''

    INDEX_FILE.write_text(html)
    print(f"\n  Rebuilt index.html with {total} collections")


def run_discovery(num_rounds: int = 3):
    """Main discovery loop."""
    print("=" * 60)
    print("  DIGITAL ARCHIVE DISCOVERY AGENT")
    print(f"  Model: {MODEL} via Ollama")
    print(f"  Rounds: {num_rounds}")
    print("=" * 60)

    # Check Ollama is running
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("\n  [error] Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    # Load existing
    existing = load_existing()
    existing_urls = get_existing_urls(existing)
    print(f"\n  Loaded {len(existing)} existing collections")
    print(f"  Known URLs: {len(existing_urls)}")

    all_discoveries = []
    if DISCOVERIES_FILE.exists():
        try:
            all_discoveries = json.loads(DISCOVERIES_FILE.read_text())
            # Snapshot for the merge-on-write below (see the race note there).
            global _baseline_discoveries
            _baseline_discoveries = list(all_discoveries)
        except Exception:
            pass

    # Run discovery rounds
    for round_num in range(num_rounds):
        print(f"\n{'—' * 60}")
        print(f"  ROUND {round_num + 1}/{num_rounds}")
        print(f"{'—' * 60}")

        # Pick a subset of domains for this round
        import random
        random.shuffle(SEARCH_DOMAINS)
        domains_this_round = SEARCH_DOMAINS[:6]

        round_candidates = []
        for i, domain in enumerate(domains_this_round):
            results = search_archives(domain, existing_urls, i + 1)
            round_candidates.extend(results)
            time.sleep(1)  # Be gentle with the LLM

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

    # Save discoveries
    # Re-read before writing. `all_discoveries` was loaded at the start of this
    # run, and a run takes minutes (LLM calls plus URL validation), so anything
    # that edited discoveries.json meanwhile would be silently reverted by a
    # blind overwrite. That is exactly what happened on 2026-08-06 10:06: a
    # maintenance pass ran mid-discovery and its work was lost. Merge instead of
    # clobber, keeping whatever is on disk as the base.
    new_entries = all_discoveries[len(_baseline_discoveries):] \
        if len(all_discoveries) >= len(_baseline_discoveries) else all_discoveries
    try:
        on_disk = json.loads(DISCOVERIES_FILE.read_text())
    except Exception:
        on_disk = all_discoveries
        new_entries = []
    known = {(c.get("url", "").rstrip("/").lower(), c.get("name", "").strip().lower())
             for c in on_disk if isinstance(c, dict)}
    added = 0
    for c in new_entries:
        key = (c.get("url", "").rstrip("/").lower(), c.get("name", "").strip().lower())
        if key[0] and key not in known:
            known.add(key)
            on_disk.append(c)
            added += 1
    all_discoveries = on_disk
    DISCOVERIES_FILE.write_text(json.dumps(all_discoveries, indent=2, ensure_ascii=False))
    print(f"\n  Merged {added} new entries into {len(all_discoveries)} on disk")
    print(f"  Saved to: {DISCOVERIES_FILE}")

    # Merge all collections and rebuild HTML
    merged = []
    # Add from original JSON files
    for f in [COLLECTIONS_FILE, BATCH2_FILE]:
        if f.exists():
            try:
                merged.extend(json.loads(f.read_text()))
            except Exception:
                pass
    # Add discoveries
    merged.extend(all_discoveries)

    # Deduplicate by (url, name). After fallback substitution, many entries from
    # the same institution legitimately share a URL but have distinct descriptive
    # metadata — keep them as separate cards so the user can still browse by topic.
    seen = set()
    deduped = []
    for c in merged:
        url = c.get("url", "").rstrip("/").lower()
        name = c.get("name", "").strip().lower()
        key = (url, name)
        if url and key not in seen:
            seen.add(key)
            deduped.append(c)

    print(f"  Total unique collections: {len(deduped)}")

    # Persist the deduped set too. Rendering a deduped list while storing the
    # un-deduped one made the file and the page disagree by ~3,000 entries, and
    # the stats on the page then described neither.
    _disc_keys = {(c.get("url", "").rstrip("/").lower(),
                   c.get("name", "").strip().lower()) for c in deduped}
    kept = [c for c in all_discoveries
            if (c.get("url", "").rstrip("/").lower(),
                c.get("name", "").strip().lower()) in _disc_keys]
    if len(kept) < len(all_discoveries):
        print(f"  Dropping {len(all_discoveries)-len(kept)} entries that duplicate "
              f"another card's title AND destination")
        DISCOVERIES_FILE.write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    # Rebuild HTML
    build_html(deduped)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(deduped)} collections in index.html")
    print(f"  {len(all_discoveries)} new discoveries this session")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_discovery(rounds)

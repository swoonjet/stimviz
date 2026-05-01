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
    """Content-aware URL validation. Returns True iff:
       - URL has acceptable shape (no query strings, reasonable depth/length)
       - Server returns 200 with a body that doesn't read as a soft-404
       - OR returns 403 (institution exists, just bot-blocking us)

    Rejects: 404, soft-404 (page-not-found in body), DNS failures, refused,
             timeouts, malformed URLs."""
    if not is_acceptable_url_shape(url):
        return False
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(8192)
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                try:
                    raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                except Exception:
                    pass
            try:
                body = raw.decode("utf-8", errors="replace")
            except Exception:
                body = ""
            # Title soft-404
            tm = _TITLE_RE.search(body)
            title = (tm.group(1) if tm else "").strip()
            if _NF_RE.search(title):
                return False
            # Cloudflare / bot challenge — treat as bot-blocked, accept (server is real)
            if any(p in title for p in ("Just a moment", "Attention Required", "Cloudflare")):
                return True
            # Body soft-404
            snippet = re.sub(r"<[^>]+>", " ", body[:3000])
            if _NF_RE.search(snippet):
                return False
            # Too short = probably a redirect target or empty page
            if len(body.strip()) < 200:
                return False
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        if e.code in (401, 403):
            return True  # bot-blocked but institution exists
        if e.code < 500:
            return False  # other 4xx — treat as broken
        return False
    except urllib.error.URLError as e:
        # All transport-level failures: be strict — reject
        return False
    except Exception:
        return False


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
    # Sort: verified-true first, via_fallback next, unknown last; then category, name.
    def _sort_key(c):
        v = c.get("verified")
        rank = 0 if v is True else (1 if v == "via_fallback" else 2)
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
        elif v == "via_fallback":
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
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 1px;
    background: #1a1a1a;
    margin: 0;
  }}

  .card {{
    background: #0f0f0f;
    padding: 0;
    display: flex;
    flex-direction: column;
    transition: background 0.2s;
  }}

  .card:hover {{ background: #141414; }}

  .card-img {{
    width: 100%;
    aspect-ratio: 4/3;
    overflow: hidden;
    position: relative;
    background: #111;
  }}

  .card-img img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.4s ease;
  }}

  .card:hover .card-img img {{ transform: scale(1.03); }}

  .card-img .placeholder {{
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #111 0%, #1a1a1a 50%, #111 100%);
  }}

  .placeholder .ph-cat {{
    font-size: 28px;
    font-weight: 600;
    color: #C9A96225;
    text-transform: uppercase;
    letter-spacing: 4px;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 12px;
  }}

  .placeholder .ph-inst {{
    font-size: 11px;
    color: #444;
    line-height: 1.5;
    max-width: 80%;
  }}

  .card-img .error-state {{
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #444;
    text-align: center;
    padding: 20px;
    line-height: 1.6;
    background: linear-gradient(135deg, #111 0%, #1a1a1a 50%, #111 100%);
  }}

  .card-body {{ padding: 24px; flex: 1; display: flex; flex-direction: column; }}

  .card-meta {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }}

  .card-category {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #C9A962;
    background: rgba(201, 169, 98, 0.1);
    padding: 3px 8px;
    border-radius: 3px;
  }}

  .card-era {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #555;
  }}

  .card-title {{
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    line-height: 1.25;
    margin-bottom: 4px;
  }}

  .card-institution {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #666;
    margin-bottom: 12px;
  }}

  .card-desc {{
    font-size: 13px;
    color: #999;
    line-height: 1.6;
    flex: 1;
    margin-bottom: 16px;
  }}

  .card-link {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #C9A962;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: color 0.2s;
  }}

  .card-link:hover {{ color: #e0c478; }}
  .card-link::after {{ content: '\\2192'; }}

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
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: 1px;
  }}
  .v-dot.live     {{ background: #6BCB77; }}
  .v-dot.fallback {{ background: #C9A962; }}
  .v-dot.unknown  {{ background: #555; }}

  .v-note {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #888;
    margin-top: 8px;
    line-height: 1.5;
  }}
  .v-note.fallback {{ color: #C9A962; }}
  .v-note.unknown  {{ color: #666; }}

  .card.unknown  {{ opacity: 0.78; }}
  .card.unknown:hover {{ opacity: 1; }}

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
    const note = noteText[v] ? `<div class="v-note ${{v}}">${{noteText[v]}}</div>` : '';
    return `
    <div class="card ${{v}}" data-category="${{c.category}}">
      <div class="card-img">
        ${{c.thumbnail
          ? `<img src="${{c.thumbnail}}" alt="${{c.name}}" loading="lazy"
               onerror="this.style.display='none'; this.parentElement.innerHTML='<div class=placeholder><div class=ph-cat>${{c.category}}</div><div class=ph-inst>${{c.institution}}</div></div>';">`
          : `<div class="placeholder"><div class="ph-cat">${{c.category}}</div><div class="ph-inst">${{c.institution}}</div></div>`
        }}
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span class="card-category">${{c.category}}</span>
          <span class="card-era">${{c.era}}</span>
        </div>
        <h2 class="card-title"><span class="v-dot ${{v}}" title="${{v}}"></span>${{c.name}}</h2>
        <p class="card-institution">${{c.institution}}</p>
        <p class="card-desc">${{c.description}}</p>
        <a class="card-link" href="${{c.url}}" target="_blank" rel="noopener">${{linkLabel[v] || linkLabel.unknown}}</a>
        ${{note}}
      </div>
    </div>
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
    DISCOVERIES_FILE.write_text(json.dumps(all_discoveries, indent=2, ensure_ascii=False))
    print(f"\n  Total new discoveries: {len(all_discoveries)}")
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

    # Deduplicate by URL
    seen = set()
    deduped = []
    for c in merged:
        url = c.get("url", "").rstrip("/").lower()
        if url and url not in seen:
            seen.add(url)
            deduped.append(c)

    print(f"  Total unique collections: {len(deduped)}")

    # Rebuild HTML
    build_html(deduped)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(deduped)} collections in index.html")
    print(f"  {len(all_discoveries)} new discoveries this session")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_discovery(rounds)

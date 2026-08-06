#!/usr/bin/env python3
"""
Shared link-checking core for stimviz.

The whole point of this module is that an HTTP status code means different
things on different hosts, so a verdict is only trustworthy when it is read in
the context of how that particular host behaves. agent/hostprobe.py learns that
context; this module applies it.

Verdicts:
  LIVE          the URL resolves to a real page
  DEAD          the URL does not exist (proven: honest 404, host's own 404
                shell, or absent from the institution's authoritative index)
  UNVERIFIABLE  cannot be determined over HTTP (bot wall, or a soft-404 host
                whose 404 shell is indistinguishable from a real page)

The old bug this replaces: agent/discover.py treated 401/403 as LIVE
("bot-blocked but institution exists"). On a host that challenges every
request, that made every invented URL look valid.
"""

import difflib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).parent.parent
PROFILE_PATH = PROJECT_DIR / "agent" / "host-profile.json"
INDEX_DIR = PROJECT_DIR / "agent" / "indexes"

LIVE, DEAD, UNVERIFIABLE = "LIVE", "DEAD", "UNVERIFIABLE"

UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

CHALLENGE_MARKERS = (
    "just a moment", "attention required", "checking your browser",
    "request verification", "access denied", "enable javascript and cookies",
    "verifying connection", "site protection", "making sure you're not a bot",
    "making sure you&#39;re not a bot", "security checkpoint", "captcha",
    "azure waf", "cf-browser-verification", "ddos-guard",
)

# Phrases that mean "not found" when they appear in a page's title or early text.
NOT_FOUND_PATTERNS = [
    r"\bpage not found\b", r"\b404 (?:error|not found|page)\b", r"\berror 404\b",
    r"\bpage you (?:are looking for|requested) (?:could not be|cannot be|was not|does not)\b",
    r"\bthis page (?:does not exist|cannot be found|isn'?t here)\b",
    r"\bno longer available\b", r"\bpage has moved\b",
    r"\bwe (?:can'?t|cannot|could not) find (?:that|the|this) page\b",
    r"\bsorry,? (?:that page|this page|the page|your page)\b",
    r"\bbroken link\b", r"\b(?:url|page) does not exist\b",
    r"\bnot found on this server\b", r"\bfile not found\b",
    r"\bno results? (?:were )?found\b", r"\bnothing (?:was )?found\b",
    r"\bthe requested (?:page|url|resource)\b",
    r"\bcollection not found\b", r"\bitem not found\b",
]
NF_RE = re.compile("|".join(NOT_FOUND_PATTERNS), re.I)

TAG_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
STRIP_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


# ---------------------------------------------------------------- fetching

def fetch(url, ua=UA_BROWSER, timeout=20, max_bytes=200000):
    """GET via curl. Returns (code, body, effective_url); code None on failure."""
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


def fetch_resilient(url, timeout=20):
    """Try a browser UA, then curl's own UA. Some WAFs (loc.gov) reject a
    spoofed browser UA but allow an honest client, so the fallback is not
    redundant."""
    code, body, eff = fetch(url, ua=UA_BROWSER, timeout=timeout)
    if code is None or code in (403, 401, 429):
        c2, b2, e2 = fetch(url, ua=None, timeout=timeout)
        if c2 is not None and (code is None or c2 not in (403, 401, 429)):
            return c2, b2, e2
    return code, body, eff


# ---------------------------------------------------------------- parsing

def visible_text(body, limit=2500):
    t = TAG_RE.sub(" ", body or "")
    t = STRIP_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()[:limit]


def title_of(body):
    m = TITLE_RE.search(body or "")
    return WS_RE.sub(" ", m.group(1)).strip()[:160] if m else ""


def looks_challenged(code, body, title=""):
    hay = (title + " " + (body or "")[:6000]).lower()
    if code in (403, 401, 429):
        return True
    return any(m in hay for m in CHALLENGE_MARKERS)


def says_not_found(title, text):
    if NF_RE.search(title or ""):
        return True
    return bool(NF_RE.search((text or "")[:2500]))


def path_depth(url):
    try:
        return len([s for s in urlparse(url).path.strip("/").split("/") if s])
    except Exception:
        return 0


def _norm_for_compare(u):
    u = (u or "").split("?")[0].split("#")[0]
    u = re.sub(r"^https?://", "", u).rstrip("/").lower()
    return re.sub(r"^www\.", "", u)


def redirect_to_error_page(orig, final):
    """True if a redirect landed on something that is an error page by name.

    Deliberately narrow. A broader "does the destination still resemble what we
    asked for?" test punishes legitimate migrations: the British Library moved
    /catalogues-and-collections to /collection, and the National Library of
    Australia moved nla.gov.au/collections to library.gov.au/discover. Both
    destinations are real and related, and both looked "unrelated" by slug.

    The genuine problem — one generic page absorbing hundreds of dead URLs, as
    bl.uk/stories does — is a property of the corpus, not of one URL, so it is
    detected in agent/catchall.py where the pattern is actually visible.
    """
    if not final or _norm_for_compare(orig) == _norm_for_compare(final):
        return False
    return bool(re.search(r"/(404|not[-_]?found|error|page[-_]?not[-_]?found)(\.\w+)?/?$",
                          urlparse(final).path, re.I))


def similarity(a, b):
    """Cheap text similarity, 0..1."""
    if not a or not b:
        return 0.0
    a, b = a[:1500], b[:1500]
    sm = difflib.SequenceMatcher(None, a, b)
    if sm.real_quick_ratio() < 0.55:
        return sm.real_quick_ratio()
    return sm.quick_ratio()


# ---------------------------------------------------------------- profile

class HostProfiles:
    def __init__(self, path=PROFILE_PATH):
        self.by_host = {}
        if Path(path).exists():
            for r in json.loads(Path(path).read_text()):
                self.by_host[r["host"]] = r

    def get(self, url):
        return self.by_host.get(urlparse(url).netloc)

    def cls(self, url):
        r = self.get(url)
        return r.get("class", "UNKNOWN") if r else "UNKNOWN"


# ------------------------------------------------- authoritative indexes

class AuthoritativeIndex:
    """Institution-published lists of real collection URLs.

    When an institution publishes its own index, that index is the ground
    truth: a URL in the same space that is absent from it does not exist. This
    is the only signal strong enough to call a bot-walled URL DEAD.
    """

    def __init__(self, index_dir=INDEX_DIR):
        self.spaces = []   # (host, path_prefix, {normalized_url: title})
        d = Path(index_dir)
        if not d.exists():
            return
        for f in sorted(d.glob("*.json")):
            try:
                spec = json.loads(f.read_text())
            except Exception:
                continue
            urls = {self.norm(u): t for u, t in (spec.get("urls") or {}).items()}
            if urls:
                self.spaces.append((spec["host"], spec.get("path_prefix", "/"), urls,
                                    spec.get("name", f.stem)))

    @staticmethod
    def norm(u):
        u = (u or "").split("?")[0].split("#")[0].replace("http://", "https://")
        if not u.endswith("/"):
            u += "/"
        return u.lower()

    def lookup(self, url):
        """Returns (verdict, index_name) or (None, None) if no index covers it."""
        p = urlparse(url)
        n = self.norm(url)
        for host, prefix, urls, name in self.spaces:
            if p.netloc != host:
                continue
            if not p.path.startswith(prefix):
                continue
            # The prefix page itself (e.g. /collections/) is the institution's
            # own browse page, not a member of the list. An index of collections
            # says nothing about whether the index page exists.
            if p.path.rstrip("/") == prefix.rstrip("/"):
                return None, None
            return (LIVE if n in urls else DEAD), name
        return None, None

    def titles_for_host(self, host, prefix="/"):
        for h, pfx, urls, name in self.spaces:
            if h == host and pfx == prefix:
                return urls
        return {}


# ---------------------------------------------------------------- verdict

def verdict_for(url, profiles, indexes=None, fetched=None):
    """Decide LIVE / DEAD / UNVERIFIABLE for one URL.

    `fetched` lets a caller pass an already-performed (code, body, eff) so a URL
    is never requested twice. Returns a dict with the verdict and its reason.
    """
    host = urlparse(url).netloc
    prof = profiles.by_host.get(host)
    hcls = (prof or {}).get("class", "UNKNOWN")

    # An institution's own index beats anything we could infer over HTTP.
    if indexes:
        v, iname = indexes.lookup(url)
        if v:
            return {"verdict": v, "reason": f"authoritative_index:{iname}",
                    "host_class": hcls, "requested": False}

    # A bot wall answers every path identically, so requesting proves nothing.
    if hcls == "BLOCKED":
        return {"verdict": UNVERIFIABLE, "reason": "host_blocks_all_requests",
                "host_class": hcls, "requested": False}

    if hcls == "UNREACHABLE":
        return {"verdict": UNVERIFIABLE, "reason": "host_unreachable",
                "host_class": hcls, "requested": False}

    code, body, eff = fetched if fetched is not None else fetch_resilient(url)

    if code is None:
        return {"verdict": UNVERIFIABLE, "reason": "request_failed",
                "host_class": hcls, "http": None, "requested": True}

    title = title_of(body)
    text = visible_text(body)

    if looks_challenged(code, body, title):
        return {"verdict": UNVERIFIABLE, "reason": "challenged", "http": code,
                "host_class": hcls, "requested": True}

    if code in (404, 410):
        return {"verdict": DEAD, "reason": f"http_{code}", "http": code,
                "host_class": hcls, "requested": True}

    if code >= 500:
        return {"verdict": UNVERIFIABLE, "reason": f"http_{code}", "http": code,
                "host_class": hcls, "requested": True}

    if code != 200:
        return {"verdict": UNVERIFIABLE, "reason": f"http_{code}", "http": code,
                "host_class": hcls, "requested": True}

    # 200 from here on.
    if says_not_found(title, text):
        return {"verdict": DEAD, "reason": "soft_404_text", "http": code,
                "host_class": hcls, "requested": True}

    # A deep path that lands on the bare host is a redirect-flavoured 404.
    if eff and path_depth(url) >= 1 and path_depth(eff) == 0:
        return {"verdict": DEAD, "reason": "redirected_to_root", "http": code,
                "host_class": hcls, "final": eff, "requested": True}

    # Redirected onto a page that is an error page by name (Getty answers 200
    # from /404.html). Broader "unrelated destination" detection lives in
    # agent/catchall.py, where the corpus makes the pattern visible.
    if eff and redirect_to_error_page(url, eff):
        return {"verdict": DEAD, "reason": "redirected_to_error_page",
                "http": code, "host_class": hcls, "final": eff, "requested": True}

    # On a host that answers 200 for everything, compare against the 404 we
    # captured during profiling.
    #
    # Body similarity is NOT usable for this, in either direction:
    #   - A JS-shell host (ufdc.ufl.edu serves 22 characters of text) returns a
    #     byte-identical shell for real and invented paths alike, so "identical
    #     to the 404" says nothing.
    #   - A big server-rendered site (www.ucalgary.ca, 1800 chars) shares so
    #     much nav/footer chrome between its 404 and its real pages that even
    #     the homepage scores 0.99 against the 404.
    # The 404 page's *title* is the discriminating signal, so use that.
    if hcls == "SOFT404" and prof:
        shell_title = (prof.get("fake_title") or "").strip().lower()
        shell_text = prof.get("fake_text") or ""
        shell_is_js_only = len(shell_text) < 400

        # A host's own root is never that host's 404, whatever it resembles.
        at_host_root = path_depth(url) == 0

        # Order matters. Check for a content-free shell FIRST: when the host
        # serves a JS shell, the title belongs to that shell too, so a matching
        # title is no more meaningful than a matching body. Europeana serves
        # "Error | Europeana" with 264 characters of text for its real homepage
        # as well as for invented paths.
        if shell_is_js_only or len(text) < 400:
            return {"verdict": UNVERIFIABLE, "reason": "spa_shell_indistinguishable",
                    "http": code, "host_class": hcls, "requested": True}

        # Host serves a substantive 404 page, so its title is discriminating.
        if not at_host_root and shell_title and title.strip().lower() == shell_title:
            return {"verdict": DEAD, "reason": "matches_host_404_title",
                    "http": code, "host_class": hcls, "requested": True}

    if len(text) < 60:
        return {"verdict": UNVERIFIABLE, "reason": "empty_body",
                "http": code, "host_class": hcls, "requested": True}

    return {"verdict": LIVE, "reason": "http_200_real_content", "http": code,
            "host_class": hcls, "final": eff if eff != url else None,
            "title": title, "requested": True}

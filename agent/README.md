# stimviz link maintenance

Live at <https://jontoews.com/stimviz/>, served by GitHub Pages from
`swoonjet/stimviz` (branch `main`, root). `index.html` is a single generated
file; `discoveries.json` is the data behind it.

## Refreshing the links

```bash
./agent/maintain.sh          # verify + repair + rebuild, ~20-40 min
./agent/maintain.sh --fresh  # ignore caches and re-check everything
python3 agent/audit_page.py --live 60   # the gate; exits non-zero on failure
```

Nothing is committed automatically. Review `git diff --stat`, then commit
`discoveries.json` + `index.html` and push.

**Confirm the deploy actually published.** Builds here take ~4 minutes and have
failed silently, leaving the site stale for hours:

```bash
gh api repos/swoonjet/stimviz/pages/builds --jq '.[0] | {status,commit,duration}'
gh api -X POST repos/swoonjet/stimviz/pages/builds   # retry a failed build
```

## Hourly discovery is disabled

`com.stimviz.discovery` is booted out **and** marked disabled, so it does not
return at login. To bring it back:

```bash
launchctl enable gui/$(id -u)/com.stimviz.discovery
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.stimviz.discovery.plist
```

Before re-enabling it, know what it costs: each run commits a multi-megabyte
`index.html`, and 1,935 such commits are why the repository is ~271 MB and the
Pages build is slow enough to time out. If it runs hourly again, consider
committing only `discoveries.json` and generating the page at deploy time.

**Never run a maintenance pass while it is loaded.** `maintain.sh` refuses to,
because the discovery run reads `discoveries.json` at the start and writes it at
the end — an overlapping pass gets silently reverted. That happened on
2026-08-06 and cost a full pass.

## Why link checking here is not just "does it return 200"

Of ~2,000 hosts in this dataset, an HTTP status code is only meaningful on about
a third. `agent/hostprobe.py` learns which is which by requesting a path that
cannot exist:

| class | share of entries | what a status means |
|---|---|---|
| `BLOCKED` | ~45% | Bot wall answers every path identically. Status proves nothing. |
| `SOFT404` | ~18% | Returns 200 for invented paths, renders the 404 in JavaScript. |
| `HONEST` | ~31% | Status is trustworthy. |
| `UNREACHABLE` | ~4% | DNS/TLS failure. |

So verdicts are `LIVE` / `DEAD` / `UNVERIFIABLE`, and **unverifiable is not
valid** — treating it as valid is the original bug that filled this index with
invented URLs (`check_url` returned `True` for 403, and a local LLM was
inventing plausible collection URLs on exactly those bot-walled hosts).
Measured against the Library of Congress's own collection index: 96% of the
loc.gov `/collections/` links pointed at slugs that never existed.

## The scripts

| script | does |
|---|---|
| `hostprobe.py` | learns each host's not-found behaviour + a usable landing page |
| `linkcheck.py` | the verdict rules, read in host context |
| `verify.py` | per-URL verdicts; serialised per host on purpose |
| `catchall.py` | corpus-level: many URLs landing on one page nobody asked for |
| `retarget.py` | matches cards to real collections via institution indexes |
| `reanchor.py` | moves unverifiable links to somewhere that works |
| `searchlink.py` | subject-search fallback instead of a generic browse page |
| `audit_page.py` | audits what `index.html` will actually serve |
| `cdxcheck.py` | Wayback CDX — the only evidence available on walled hosts |
| `indexes/` | institution-published collection lists (LoC: 583 collections) |
| `discover.py` | the LLM discovery agent + `build_html()` |

## Things that look like bugs but are not

- **Verify is slow and single-threaded per host.** Deliberate. Parallel requests
  to one host provoke rate-limiting that is indistinguishable from bot-blocking,
  which corrupts verdicts.
- **Most links are `UNVERIFIABLE`.** Expected: ~63% of entries sit on hosts where
  HTTP cannot answer the question. They are pointed at destinations that do
  answer, and the card says what it opens.
- **`retarget.py` only matches a few dozen cards.** It requires a high score
  *and* a clear margin over the runner-up. Looser settings confidently produced
  "World War II Posters" → *World War I Posters*.
- **Card counts drop after a pass.** Re-anchoring makes same-title cards share a
  destination, at which point they are true duplicates and are dropped.

## Do not

- Do not widen the redirect heuristic in `linkcheck.py` to "destination no longer
  resembles the request". It condemns real migrations — the British Library moved
  `/catalogues-and-collections` to `/collection`. Use `catchall.py` instead.
- Do not compare page bodies to a host's 404 page to decide liveness. A JS-shell
  host serves an identical shell for real and invented paths, and a large
  server-rendered site shares so much nav chrome that its own homepage scores
  0.99 against its 404. Compare the 404 *title*, and only when the 404 has
  substance.
- Do not trust `discoveries.before-reanchor.json` as a pre-change backup unless
  it was written before mutation. `git show HEAD:discoveries.json` is authoritative.

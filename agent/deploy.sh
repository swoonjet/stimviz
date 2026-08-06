#!/bin/bash
# Push and actually confirm the site published.
#
#   ./agent/deploy.sh "commit message"     commit tracked changes, push, verify
#   ./agent/deploy.sh --verify-only        just check that live matches local
#
# Why this exists: on this repository, the build GitHub triggers from a push
# frequently fails instantly (status "errored", duration 0s) while a build
# requested through the API for the very same commit succeeds in ~250s. The push
# reports success either way, so the site can sit stale for hours with no signal.
# Observed repeatedly on 2026-08-06. This script pushes, then requests a build,
# waits for it, and byte-compares the live page against the local one.

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO="swoonjet/stimviz"
URL="https://jontoews.com/stimviz/"

if [ "$1" != "--verify-only" ]; then
  MSG="${1:?usage: ./agent/deploy.sh \"commit message\"  |  --verify-only}"

  echo "==> auditing the built page before publishing it"
  python3 agent/audit_page.py || {
    echo "!! audit failed — not deploying. Fix the findings or run maintain.sh."
    exit 1
  }

  if [ -n "$(git status --porcelain discoveries.json index.html)" ]; then
    git add discoveries.json index.html
    git commit -q -m "$MSG"
    echo "==> committed: $(git log --oneline -1)"
  else
    echo "==> no changes to discoveries.json / index.html"
  fi

  git push origin main
fi

LOCAL_BYTES=$(wc -c < index.html | tr -d ' ')
COMMIT=$(git rev-parse HEAD | cut -c1-7)
echo "==> local index.html: $LOCAL_BYTES bytes (commit $COMMIT)"

# Ask for a build explicitly rather than trusting the push-triggered one.
echo "==> requesting a Pages build"
gh api -X POST "repos/$REPO/pages/builds" --jq '.status' || true

echo "==> waiting for the build (takes ~4 minutes)"
for i in $(seq 1 40); do
  STATUS=$(gh api "repos/$REPO/pages/builds/latest" --jq '.status' 2>/dev/null || echo "?")
  [ "$STATUS" = "building" ] || [ "$STATUS" = "queued" ] || break
  sleep 20
done
gh api "repos/$REPO/pages/builds/latest" \
  --jq '"    build: \(.status)  \(.commit[0:7])  \((.duration/1000)|floor)s  \(.error.message // "")"'

if [ "$(gh api "repos/$REPO/pages/builds/latest" --jq '.status')" != "built" ]; then
  echo "!! build did not succeed. Retry with:"
  echo "     gh api -X POST repos/$REPO/pages/builds"
  exit 1
fi

echo "==> confirming the live page matches local"
for i in $(seq 1 15); do
  REMOTE=$(curl -s -o /dev/null -w "%{size_download}" "$URL")
  if [ "$REMOTE" = "$LOCAL_BYTES" ]; then
    echo "    PUBLISHED — live matches local ($REMOTE bytes)"
    exit 0
  fi
  echo "    live=$REMOTE want=$LOCAL_BYTES (CDN catching up, check $i)"
  sleep 20
done

echo "!! live page still does not match local. The build succeeded, so this is"
echo "   likely CDN caching; re-check with:  ./agent/deploy.sh --verify-only"
exit 1

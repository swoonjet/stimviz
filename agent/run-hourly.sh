#!/bin/bash
# Hourly discovery agent — runs via cron, commits and pushes new finds
set -e

REPO_DIR="/Users/jontoewsinterceptgroup.com/Projects/digital-collections"
LOG_FILE="$REPO_DIR/agent/agent.log"
LOCK_FILE="$REPO_DIR/agent/.running"

# Prevent overlapping runs
if [ -f "$LOCK_FILE" ]; then
  pid=$(cat "$LOCK_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "$(date): Agent already running (pid $pid), skipping" >> "$LOG_FILE"
    exit 0
  fi
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

echo "$(date): Starting discovery run" >> "$LOG_FILE"

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "$(date): Ollama not running, attempting start" >> "$LOG_FILE"
  ollama serve &
  sleep 5
fi

cd "$REPO_DIR"

# Count before
BEFORE=$(python3 -c "import json; print(len(json.load(open('discoveries.json'))))" 2>/dev/null || echo 0)

# Run 2 rounds of discovery
python3 agent/discover.py 2 >> "$LOG_FILE" 2>&1

# Count after
AFTER=$(python3 -c "import json; print(len(json.load(open('discoveries.json'))))" 2>/dev/null || echo 0)
NEW=$((AFTER - BEFORE))

if [ "$NEW" -gt 0 ]; then
  echo "$(date): Found $NEW new collections, committing" >> "$LOG_FILE"

  # Get total count from the HTML title
  TOTAL=$(grep -o '[0-9]* Digital Archives' index.html | head -1 | grep -o '[0-9]*')

  git add index.html discoveries.json
  git commit -m "Add $NEW new archives ($TOTAL total) — agent discovery $(date +%Y-%m-%d\ %H:%M)"
  git push origin main

  echo "$(date): Pushed $NEW new collections (total: $TOTAL)" >> "$LOG_FILE"
else
  echo "$(date): No new collections found this run" >> "$LOG_FILE"
fi

echo "$(date): Done" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

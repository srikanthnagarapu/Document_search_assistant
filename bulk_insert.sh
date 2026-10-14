#!/usr/bin/env bash
set -euo pipefail

# Path to bulk payload (inside the project folder)
PAYLOAD_FILE="$(pwd)/bulk_payload.json"

if [[ ! -f "$PAYLOAD_FILE" ]]; then
  echo "Bulk payload file not found: $PAYLOAD_FILE"
  exit 1
fi

# Iterate over each document and POST it
python3 - <<'PY'
import json, subprocess, os, sys
payload_path = os.getenv('PAYLOAD_FILE')
with open(payload_path, 'r') as f:
    docs = json.load(f)
for doc in docs:
    # Send POST request
    result = subprocess.run([
        'curl', '-s', '-X', 'POST', 'http://127.0.0.1:8000/index',
        '-H', 'Content-Type: application/json',
        '--data-binary', json.dumps(doc)
    ], capture_output=True, text=True)
    print('POSTed:', doc.get('title'))
    print('Response:', result.stdout.strip())
PY

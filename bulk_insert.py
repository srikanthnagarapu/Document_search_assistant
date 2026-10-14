#!/usr/bin/env python3
import json, subprocess, os, sys

# Determine project root (where this script lives)
script_dir = os.path.abspath(os.path.dirname(__file__))
payload_path = os.path.join(script_dir, "bulk_payload.json")

if not os.path.isfile(payload_path):
    print(f"Payload file not found: {payload_path}")
    sys.exit(1)

with open(payload_path, "r") as f:
    docs = json.load(f)

for doc in docs:
    # Prepare curl command
    cmd = [
        "curl", "-s", "-X", "POST", "http://127.0.0.1:8000/index",
        "-H", "Content-Type: application/json",
        "--data-binary", json.dumps(doc)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("POSTed:", doc.get("title"))
    print("Response:", result.stdout.strip())

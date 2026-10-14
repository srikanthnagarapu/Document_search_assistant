#!/usr/bin/env python3
import json, subprocess, os, sys

# Resolve the path to the bulk payload (same directory as this script)
script_dir = os.path.abspath(os.path.dirname(__file__))
payload_path = os.path.join(script_dir, "bulk_payload.json")

if not os.path.isfile(payload_path):
    print(f"Payload file not found: {payload_path}", file=sys.stderr)
    sys.exit(1)

with open(payload_path, "r") as f:
    docs = json.load(f)

for doc in docs:
    # POST each document to the FastAPI /index endpoint
    result = subprocess.run([
        "curl", "-s", "-X", "POST", "http://127.0.0.1:8000/index",
        "-H", "Content-Type: application/json",
        "--data-binary", json.dumps(doc)
    ], capture_output=True, text=True)
    print("POSTed:", doc.get("title"))
    print("Response:", result.stdout.strip())

print("All documents processed.")

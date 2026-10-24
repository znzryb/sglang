"""Tiny HTTP helper so scripts don't depend on curl. usage: http.py GET url | http.py POST url json"""
import json, sys, urllib.request
method, url = sys.argv[1], sys.argv[2]
data = sys.argv[3].encode() if len(sys.argv) > 3 else None
req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        print(r.read().decode()); sys.exit(0)
except Exception as e:
    print(f"HTTP_ERROR {e}", file=sys.stderr); sys.exit(1)

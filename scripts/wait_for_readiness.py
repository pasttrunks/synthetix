#!/usr/bin/env python3
"""
Wait for Synthetix API server readiness on http://127.0.0.1:8000/health
"""
import time
import sys
import requests

def main():
    url = "http://127.0.0.1:8000/health"
    timeout_seconds = 60
    start_time = time.time()
    
    print(f"Waiting for server readiness at {url}...")
    while time.time() - start_time < timeout_seconds:
        try:
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                print("Server is ready!")
                sys.exit(0)
        except Exception:
            pass
        time.sleep(1)
        
    print(f"Timed out after {timeout_seconds} seconds waiting for server readiness at {url}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()

"""
Test script for verifying live OpenAQ API query and location IDs.
"""

import os
import re
import requests
from pathlib import Path

def test_locations():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    key = ""
    if env_path.exists():
        m = re.search(r"OPENAQ_API_KEY\s*=\s*([^\r\n]+)", env_path.read_text(encoding="utf-8"))
        if m:
            key = m.group(1).strip()
    
    headers = {"X-API-Key": key}
    locations = [
        (6932, "Alipur"),
        (235, "Anand Vihar"),
        (17, "RK Puram"),
        (50, "Punjabi Bagh"),
        (5613, "ITO"),
        (6356, "Pusa"),
        (6358, "Mandir Marg"),
        (6931, "Dwarka"),
        (6938, "Vivek Vihar"),
        (6960, "Patparganj"),
        (8472, "Bawana"),
        (10486, "Mundka"),
        (10831, "Rohini"),
    ]
    
    print(f"Testing {len(locations)} locations with OpenAQ API v3...")
    for lid, name in locations:
        url = f"https://api.openaq.org/v3/locations/{lid}/latest"
        try:
            r = requests.get(url, headers=headers, timeout=8)
            res = r.json().get("results", []) if r.status_code == 200 else []
            print(f"Location {lid:5d} ({name:15s}): status={r.status_code}, readings={len(res)}")
        except Exception as e:
            print(f"Location {lid:5d} ({name:15s}): error={e}")

if __name__ == "__main__":
    test_locations()

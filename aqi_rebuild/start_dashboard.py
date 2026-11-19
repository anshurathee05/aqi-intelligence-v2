"""
Launch helper for the AQI Intelligence Streamlit Dashboard.
"""

import sys
import subprocess
from pathlib import Path

def main():
    dashboard_app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_app)]
    print(f"Launching AQI Intelligence Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

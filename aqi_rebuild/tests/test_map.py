"""
Test Suite: Delhi Map Rendering Verification.
Verifies that create_delhi_map() functions without AttributeError on modern Plotly (scatter_map).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.config.settings import LATEST_PREDICTIONS_JSON
from aqi_rebuild.dashboard.components import create_delhi_map

def test_map_render():
    stations = []
    if LATEST_PREDICTIONS_JSON.exists():
        with open(LATEST_PREDICTIONS_JSON, "r", encoding="utf-8") as f:
            stations = json.load(f).get("stations", [])
    
    # Test with real station data
    fig = create_delhi_map(stations)
    assert fig is not None, "Map figure should not be None"
    assert len(fig.data) > 0, "Map figure should contain traces"

    # Test with empty data
    empty_fig = create_delhi_map([])
    assert empty_fig is not None

    print(f"✅ Map test passed: Rendered {len(stations)} stations successfully with Plotly.")

if __name__ == "__main__":
    test_map_render()

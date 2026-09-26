from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_methods_note_tables_regenerate_in_quick_mode(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "paper" / "reproduce.py"),
            "--quick",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    produced = sorted(item.name for item in tmp_path.glob("*.csv"))
    assert produced == [
        "table1_detector_calibration.csv",
        "table2_fusion_coverage.csv",
        "table3_small_sample_intervals.csv",
        "table4_shared_control_concordance.csv",
        "table5_age_acceleration_leakage.csv",
        "table6_surrogate_precision.csv",
    ]

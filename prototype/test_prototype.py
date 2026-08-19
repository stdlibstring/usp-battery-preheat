from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
OUTPUT = ROOT / "output"
EXE = BUILD / "preheat_prototype.exe"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)


def load_summary() -> dict[str, float]:
    with (OUTPUT / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    run(["gcc", "-Wall", "-Wextra", "-Werror", "-O2", "-std=c11", str(ROOT / "preheat_prototype.c"), "-lm", "-o", str(EXE)])
    assert "SELF_TEST_PASS" in run([str(EXE), "--self-test"]).stdout
    result = run([str(EXE), str(ROOT / "example_route_synthetic.csv"), str(OUTPUT), "0", "65"])
    assert "ALL CONSTRAINTS SATISFIED" in result.stdout

    summary = load_summary()
    assert 20.0 <= summary["end_temp_c"] <= 25.0
    assert summary["end_soc_pct"] >= 10.0
    assert 0.0 <= summary["start_distance_km"] <= summary["route_distance_km"]
    assert summary["heat_energy_kwh"] > 0.0
    assert summary["charge_time_s"] >= 0.0
    assert summary["feasible_candidates"] >= 1

    with (OUTPUT / "candidate_sweep.csv").open("r", encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == int(summary["feasible_candidates"])
    assert max(float(row["score_50"]) for row in candidates) == summary["score_50"]

    with (OUTPUT / "trajectory.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 2
    times = [float(row["time_s"]) for row in rows]
    assert max(b - a for a, b in zip(times, times[1:])) <= 0.500001
    assert abs(float(rows[-1]["remaining_km"])) < 1e-8
    assert {float(row["heating_kw"]) for row in rows} <= {0.0, 6.0}

    run([sys.executable, str(ROOT / "plot_results.py"), str(OUTPUT)])
    for filename in [
        "battery_temp.png",
        "battery_soc.png",
        "discharge_current.png",
        "heating_power.png",
        "remaining_dist.png",
    ]:
        path = OUTPUT / filename
        assert path.stat().st_size > 10_000
        with Image.open(path) as image:
            assert image.size == (1200, 700)

    print("ALL_TESTS_PASS")
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

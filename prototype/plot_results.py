from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 700
MARGIN_LEFT = 105
MARGIN_RIGHT = 40
MARGIN_TOP = 72
MARGIN_BOTTOM = 92


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_LABEL = load_font(22)
FONT_TICK = load_font(18)
FONT_NOTE = load_font(17)


def load_trajectory(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("trajectory.csv is empty")
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def load_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def nice_bounds(
    values: list[float], include: tuple[float, ...] = (), zero_floor: bool = False
) -> tuple[float, float]:
    low = min([*values, *include])
    high = max([*values, *include])
    if math.isclose(low, high):
        pad = max(1.0, abs(low) * 0.1)
    else:
        pad = (high - low) * 0.08
    lower = low - pad
    if zero_floor and low >= 0.0:
        lower = 0.0
    return lower, high + pad


def draw_chart(
    output: Path,
    times_s: list[float],
    values: list[float],
    title: str,
    y_label: str,
    color: tuple[int, int, int],
    preheat_start_s: float,
    target_band: tuple[float, float] | None = None,
    reference_lines: tuple[float, ...] = (),
    zero_floor: bool = False,
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    x0, y0 = MARGIN_LEFT, MARGIN_TOP
    x1, y1 = WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM
    includes = (*target_band, *reference_lines) if target_band else reference_lines
    y_min, y_max = nice_bounds(values, includes, zero_floor=zero_floor)
    t_min, t_max = min(times_s), max(times_s)

    def px(t: float) -> float:
        return x0 + (t - t_min) / max(1e-12, t_max - t_min) * (x1 - x0)

    def py(v: float) -> float:
        return y1 - (v - y_min) / max(1e-12, y_max - y_min) * (y1 - y0)

    if target_band:
        band_low, band_high = target_band
        draw.rectangle((x0, py(band_high), x1, py(band_low)), fill=(229, 247, 232))

    for i in range(6):
        x = x0 + i / 5 * (x1 - x0)
        t_minutes = (t_min + i / 5 * (t_max - t_min)) / 60.0
        draw.line((x, y0, x, y1), fill=(225, 229, 235), width=1)
        label = f"{t_minutes:.1f}"
        bbox = draw.textbbox((0, 0), label, font=FONT_TICK)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y1 + 14), label, fill=(60, 65, 75), font=FONT_TICK)

    for i in range(6):
        y = y1 - i / 5 * (y1 - y0)
        value = y_min + i / 5 * (y_max - y_min)
        draw.line((x0, y, x1, y), fill=(225, 229, 235), width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=FONT_TICK)
        draw.text((x0 - 12 - (bbox[2] - bbox[0]), y - 10), label, fill=(60, 65, 75), font=FONT_TICK)

    if target_band:
        for target in target_band:
            draw.line((x0, py(target), x1, py(target)), fill=(55, 150, 78), width=2)

    for reference in reference_lines:
        y = py(reference)
        for dash_x in range(x0, x1, 18):
            draw.line((dash_x, y, min(dash_x + 10, x1), y), fill=(170, 75, 65), width=2)

    preheat_x = px(preheat_start_s)
    for dash_y in range(y0, y1, 18):
        draw.line((preheat_x, dash_y, preheat_x, min(dash_y + 10, y1)), fill=(192, 57, 162), width=3)

    max_points = 3500
    stride = max(1, len(times_s) // max_points)
    indices = list(range(0, len(times_s), stride))
    if indices[-1] != len(times_s) - 1:
        indices.append(len(times_s) - 1)
    points = [(px(times_s[i]), py(values[i])) for i in indices]
    draw.line(points, fill=color, width=4, joint="curve")

    draw.rectangle((x0, y0, x1, y1), outline=(45, 50, 60), width=2)
    draw.text((MARGIN_LEFT, 20), title, fill=(25, 30, 40), font=FONT_TITLE)
    x_label = "Driving time (min)"
    bbox = draw.textbbox((0, 0), x_label, font=FONT_LABEL)
    draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, HEIGHT - 48), x_label, fill=(35, 40, 50), font=FONT_LABEL)
    label_bbox = draw.textbbox((0, 0), y_label, font=FONT_LABEL)
    label_w = label_bbox[2] - label_bbox[0]
    label_h = label_bbox[3] - label_bbox[1]
    rotated = Image.new("RGBA", (label_w + 12, label_h + 12), (255, 255, 255, 0))
    rotated_draw = ImageDraw.Draw(rotated)
    rotated_draw.text((6, 6 - label_bbox[1]), y_label, fill=(35, 40, 50), font=FONT_LABEL)
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (16, int((y0 + y1 - rotated.height) / 2)), rotated)
    note = f"Preheat starts at {preheat_start_s / 60.0:.2f} min"
    draw.text((x1 - 300, y0 + 12), note, fill=(150, 40, 125), font=FONT_NOTE)
    image.save(output, format="PNG", optimize=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} OUTPUT_DIR", file=sys.stderr)
        return 2
    output_dir = Path(sys.argv[1])
    data = load_trajectory(output_dir / "trajectory.csv")
    summary = load_summary(output_dir / "summary.csv")
    times = data["time_s"]
    start_s = summary["start_time_s"]

    charts = [
        ("battery_temp.png", data["temp_c"], "Battery temperature", "Temperature (C)", (31, 99, 210), (20.0, 25.0), (), False),
        ("battery_soc.png", data["soc_pct"], "Battery SOC", "SOC (%)", (35, 142, 92), None, (10.0,), True),
        ("discharge_current.png", data["current_a"], "Battery discharge current", "Current (A)", (226, 113, 29), None, (), True),
        ("heating_power.png", data["heating_kw"], "Battery heating power", "Power (kW)", (128, 74, 180), None, (), True),
        ("remaining_dist.png", data["remaining_km"], "Remaining driving distance", "Distance (km)", (20, 142, 158), None, (), True),
    ]
    for filename, values, title, label, color, band, refs, zero_floor in charts:
        draw_chart(
            output_dir / filename,
            times,
            values,
            title,
            label,
            color,
            start_s,
            band,
            refs,
            zero_floor,
        )
    print(f"Wrote {len(charts)} PNG charts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

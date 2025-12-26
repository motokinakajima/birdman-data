import argparse
import csv
import datetime
import platform
import sys
from pathlib import Path


def read_spanwise_distribution(file_path):
    """
    Read spanwise distribution data from XFLR5 wing output file.
    
    Returns:
        y_spans: list of spanwise positions [m]
        chords: list of local chord lengths [m]
        cls: list of local lift coefficients [-]
        meta: dict with metadata (plane_name, polar_name, alpha, qinf, CL_total)
    """
    y_spans = []
    chords = []
    cls = []
    meta = {
        "plane_name": None,
        "polar_name": None,
        "alpha": None,
        "qinf": None,
        "CL_total": None,
    }

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Extract metadata from header
    for line in lines[:20]:
        if line.startswith("QInf"):
            try:
                meta["qinf"] = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if line.startswith("Alpha"):
            try:
                meta["alpha"] = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass
        if line.startswith("CL"):
            try:
                meta["CL_total"] = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass

    # Find the line index after header row "y-span  Chord  Ai  Cl ..."
    data_start = None
    for i, line in enumerate(lines):
        if "y-span" in line and "Chord" in line and "Cl" in line:
            data_start = i + 1
            # Also extract plane name from line 2 (index 2)
            if len(lines) > 2:
                meta["plane_name"] = lines[2].strip()
            if len(lines) > 3:
                meta["polar_name"] = lines[3].strip()
            break

    if data_start is None:
        raise ValueError("Could not find spanwise data header in file")

    # Parse numerical data rows
    for line in lines[data_start:]:
        cols = line.strip().split()
        if len(cols) < 4:
            continue
        try:
            y_span = float(cols[0])
            chord = float(cols[1])
            cl = float(cols[3])  # Cl is column index 3
        except ValueError:
            continue

        y_spans.append(y_span)
        chords.append(chord)
        cls.append(cl)

    return y_spans, chords, cls, meta


def compute_lift_distribution(y_spans, chords, cls, air_density, velocity):
    """
    Compute lift force per unit span at each spanwise location.
    
    L'(y) = 0.5 * rho * V^2 * c(y) * Cl(y)  [N/m]
    
    Args:
        y_spans: list of spanwise positions [m]
        chords: list of local chord lengths [m]
        cls: list of local lift coefficients [-]
        air_density: air density [kg/m^3]
        velocity: freestream velocity [m/s]
    
    Returns:
        lift_per_span: list of lift force per unit span [N/m]
    """
    q = 0.5 * air_density * velocity * velocity
    lift_per_span = [q * c * cl for c, cl in zip(chords, cls)]
    return lift_per_span


def write_lift_distribution(output_path, y_spans, chords, cls, lift_per_span, meta, air_density, velocity):
    """
    Write lift force distribution to CSV file.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Metadata header
        f.write("# XFLR5 Lift Distribution Output\n")
        f.write(f"# Created: {datetime.datetime.now().isoformat()}\n")
        f.write(f"# Plane name: {meta.get('plane_name')}\n")
        f.write(f"# Polar name: {meta.get('polar_name')}\n")
        f.write(f"# Alpha [deg]: {meta.get('alpha')}\n")
        f.write(f"# Freestream speed [m/s]: {velocity}\n")
        f.write(f"# Air density [kg/m^3]: {air_density}\n")
        f.write(f"# Total CL [-]: {meta.get('CL_total')}\n")
        f.write(f"# Rows: {len(y_spans)}\n")
        f.write(f"# Python version: {sys.version.split()[0]}\n")
        f.write(f"# OS: {platform.platform()}\n")
        f.write("# Formula: L' [N/m] = 0.5 * rho [kg/m^3] * V^2 [m^2/s^2] * c [m] * Cl [-]\n")
        f.write("# ----------------------------------------------------------------------\n")

        writer = csv.writer(f)
        writer.writerow([
            "y_span [m]",
            "chord [m]",
            "Cl [-]",
            "lift_per_span [N/m]"
        ])

        for y, c, cl, L in zip(y_spans, chords, cls, lift_per_span):
            writer.writerow([y, c, cl, L])


def read_polar(file_path):
    alphas = []
    cls = []
    speeds = []
    meta = {
        "plane_name": None,
        "polar_name": None,
        "fstream_speed_header": None
    }

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # メタ情報抽出
    for line in lines[:20]:
        if "Plane name" in line:
            meta["plane_name"] = line.split(":")[1].strip()
        if "Polar name" in line:
            meta["polar_name"] = line.split(":")[1].strip()
        if "Freestream speed" in line:
            try:
                meta["fstream_speed_header"] = float(line.split(":")[1].split()[0])
            except:
                pass

    # 数値行（alpha, CL, QInf）抽出
    for line in lines:
        cols = line.strip().split()
        if len(cols) < 12:
            continue
        try:
            alpha = float(cols[0])
            CL = float(cols[2])
            QInf = float(cols[11])
        except:
            continue

        alphas.append(alpha)
        cls.append(CL)
        speeds.append(QInf)

    return alphas, cls, speeds, meta


def compute_lift(wing_area, air_density, speeds, cls):
    lifts = []
    for V, CL in zip(speeds, cls):
        q = 0.5 * air_density * V * V
        lifts.append(q * wing_area * CL)
    return lifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--wing_area", type=float, required=True)
    parser.add_argument("--freestream_speed", type=float, default=None)
    parser.add_argument("--air_density", type=float, default=1.225)
    parser.add_argument("--output", default="lift_output.csv")

    args = parser.parse_args()

    alphas, cls, speeds, meta = read_polar(args.file_path)

    # freestream_speed が未指定なら XFLR5 のヘッダ値を使用
    if args.freestream_speed is not None:
        speeds = [args.freestream_speed for _ in speeds]
        used_speed = args.freestream_speed
    else:
        used_speed = meta["fstream_speed_header"]

    lifts = compute_lift(
        wing_area=args.wing_area,
        air_density=args.air_density,
        speeds=speeds,
        cls=cls
    )

    with open(args.output, "w", newline="", encoding="utf-8") as f:

        # ---- 再現性用のメタ情報（単位付き） ----
        f.write("# XFLR5 Polar Lift Calculation Output\n")
        f.write(f"# Created: {datetime.datetime.now().isoformat()}\n")
        f.write(f"# Source file: {args.file_path}\n")
        f.write(f"# Plane name: {meta['plane_name']}\n")
        f.write(f"# Polar name: {meta['polar_name']}\n")
        f.write(f"# Freestream speed (header) [m/s]: {meta['fstream_speed_header']}\n")
        f.write(f"# Freestream speed (used) [m/s]: {used_speed}\n")
        f.write(f"# Wing area [m^2]: {args.wing_area}\n")
        f.write(f"# Air density [kg/m^3]: {args.air_density}\n")
        f.write(f"# Rows: {len(alphas)}\n")
        f.write(f"# Python version: {sys.version.split()[0]}\n")
        f.write(f"# OS: {platform.platform()}\n")
        f.write("# Formula: L [N] = 0.5 * rho [kg/m^3] * V^2 [m^2/s^2] * S [m^2] * CL [-]\n")
        f.write("# ----------------------------------------------------------------------\n")

        writer = csv.writer(f)

        # ---- 単位付きの列ヘッダ ----
        writer.writerow([
            "alpha_deg [deg]",
            "CL [-]",
            "V_mps [m/s]",
            "Lift_N [N]"
        ])

        for a, cl, v, L in zip(alphas, cls, speeds, lifts):
            writer.writerow([a, cl, v, L])

    print(f"出力完了: {args.output}")


def main_distribution():
    """
    Command-line interface for computing lift force distribution along the span.
    """
    parser = argparse.ArgumentParser(
        description="Compute lift force distribution from XFLR5 spanwise output"
    )
    parser.add_argument("file_path", help="Path to XFLR5 wing output file (e.g., MainWing_a=11.00_v=8.50ms.txt)")
    parser.add_argument("--air_density", type=float, default=1.225, help="Air density [kg/m^3] (default: 1.225)")
    parser.add_argument("--velocity", type=float, default=None, help="Override freestream velocity [m/s] (default: use QInf from file)")
    parser.add_argument("--output", default="lift_distribution_output.csv", help="Output CSV file path")

    args = parser.parse_args()

    # Read spanwise data
    y_spans, chords, cls, meta = read_spanwise_distribution(args.file_path)

    # Use velocity from file if not overridden
    velocity = args.velocity if args.velocity is not None else meta["qinf"]
    if velocity is None:
        raise ValueError("Could not determine velocity. Please specify --velocity or ensure QInf is in the file.")

    # Compute lift distribution
    lift_per_span = compute_lift_distribution(
        y_spans=y_spans,
        chords=chords,
        cls=cls,
        air_density=args.air_density,
        velocity=velocity
    )

    # Write output
    write_lift_distribution(
        output_path=args.output,
        y_spans=y_spans,
        chords=chords,
        cls=cls,
        lift_per_span=lift_per_span,
        meta=meta,
        air_density=args.air_density,
        velocity=velocity
    )

    print(f"出力完了: {args.output}")
    print(f"  スパン位置数: {len(y_spans)}")
    print(f"  使用速度: {velocity} m/s")


if __name__ == "__main__":
    main()
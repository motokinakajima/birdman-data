import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_rib_forces_csv(path: str) -> Tuple[List[float], List[float], float, float]:
    """
    Read rib positions and forces from CSV produced by compute_rib_force.py.
    Returns (positions_m, forces_N, wing_tip_left_m, wing_tip_right_m).
    If actual wing tips are not present in header, infers from positions.
    """
    positions: List[float] = []
    forces: List[float] = []
    wing_tip_left = None
    wing_tip_right = None

    tip_pattern = re.compile(r"Actual wing span \[m\]:\s*([-+\d\.]+) to ([-+\d\.]+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                m = tip_pattern.search(line)
                if m:
                    wing_tip_left = float(m.group(1))
                    wing_tip_right = float(m.group(2))
                continue
            if "rib_position" in line:
                # header line
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                positions.append(float(parts[0]))
                forces.append(float(parts[1]))
            except ValueError:
                continue

    if not positions:
        raise ValueError("No rib force data found in CSV.")

    if wing_tip_left is None or wing_tip_right is None:
        wing_tip_left = min(positions)
        wing_tip_right = max(positions)

    return positions, forces, wing_tip_left, wing_tip_right


def compute_tributary_spans(positions: List[float], tip_left: float, tip_right: float) -> List[float]:
    """
    Compute tributary span length for each rib using midpoint domains,
    matching the method in compute_rib_force.py.
    """
    n = len(positions)
    spans: List[float] = []
    for i, y in enumerate(positions):
        if i == 0:
            left = tip_left
        else:
            left = 0.5 * (positions[i - 1] + positions[i])
        if i == n - 1:
            right = tip_right
        else:
            right = 0.5 * (positions[i] + positions[i + 1])
        spans.append(max(0.0, right - left))
    return spans


def adjust_forces_uniform(positions: List[float], forces: List[float], tip_left: float, tip_right: float,
                          pipe_mass_g: float, factor: float = 2.0, g: float = 9.80665) -> Tuple[List[float], float]:
    """
    Subtract a uniform distributed load equal to factor * (pipe_mass_g * g) over the span.
    Returns (adjusted_forces, w_N_per_m).
    """
    span = tip_right - tip_left
    if span <= 0:
        raise ValueError("Invalid wing span: right tip must be greater than left tip.")

    total_weight_N = (pipe_mass_g / 1000.0) * g * factor
    w = total_weight_N / span  # N/m uniformly across span

    spans = compute_tributary_spans(positions, tip_left, tip_right)
    adjusted = [f - w * s for f, s in zip(forces, spans)]
    return adjusted, w


def write_adjusted_csv(path: str, positions: List[float], forces_before: List[float], 
                       forces_after: List[float], w_N_per_m: float,
                       tip_left: float, tip_right: float):
    total_before = sum(forces_before)
    total_after = sum(forces_after)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Rib forces adjusted by uniform pipe weight\n")
        f.write(f"# Actual wing span [m]: {tip_left:.4f} to {tip_right:.4f}\n")
        f.write(f"# Uniform load subtracted w [N/m]: {w_N_per_m:.6f}\n")
        f.write(f"# Total before [N]: {total_before:.6f}\n")
        f.write(f"# Total after  [N]: {total_after:.6f}\n")
        f.write("rib_position [m],force_before [N],force_after [N]\n")
        for y, fb, fa in zip(positions, forces_before, forces_after):
            f.write(f"{y},{fb:.6f},{fa:.6f}\n")


def plot_before_after(positions: List[float], forces_before: List[float], forces_after: List[float], png_path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    plt.plot(positions, forces_before, label="Before", marker="o", linewidth=1.5)
    plt.plot(positions, forces_after, label="After", marker="o", linewidth=1.5)
    plt.xlabel("rib position y [m]")
    plt.ylabel("force [N]")
    plt.title("Rib forces: before vs after uniform pipe load subtraction")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Subtract uniform pipe weight from rib forces (by tributary span)")
    parser.add_argument("--rib_forces", required=True, help="CSV from compute_rib_force.py")
    parser.add_argument("--pipe_mass_g", type=float, required=True, help="Pipe mass in grams")
    parser.add_argument("--factor", type=float, default=2.0, help="Multiply the pipe weight by this factor (default 2.0)")
    parser.add_argument("--gravity", type=float, default=9.80665, help="Gravity m/s^2 (default 9.80665)")
    parser.add_argument("--output_csv", default="rib_forces_adjusted.csv", help="Output CSV path")
    parser.add_argument("--output_png", default="rib_forces_before_after.png", help="Output PNG path")

    args = parser.parse_args()

    positions, forces, tip_left, tip_right = read_rib_forces_csv(args.rib_forces)
    adjusted, w = adjust_forces_uniform(positions, forces, tip_left, tip_right, args.pipe_mass_g, args.factor, args.gravity)

    write_adjusted_csv(args.output_csv, positions, forces, adjusted, w, tip_left, tip_right)
    plot_before_after(positions, forces, adjusted, args.output_png)

    print(f"Uniform load w = {w:.6f} N/m over span [{tip_left:.3f}, {tip_right:.3f}] m")
    print(f"Wrote {args.output_csv} and {args.output_png}")


if __name__ == "__main__":
    main()

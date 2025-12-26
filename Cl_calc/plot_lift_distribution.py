import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def piecewise_linear_interp(y_data: List[float], lift_data: List[float], y_query: float) -> float:
    """
    Piecewise linear interpolation of lift distribution.
    Given y_query, returns interpolated lift value.
    """
    return np.interp(y_query, y_data, lift_data)


def parse_main_wing_lift(file_path: str, air_density: float = 1.225, velocity: float = None) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    Parse XFLR5 Main Wing section and compute lift force per unit span.
    
    Returns:
        y_span: spanwise positions [m]
        chord: local chord lengths [m]
        cl_vals: local lift coefficients [-]
        lift_per_span: lift force per unit span [N/m]
        velocity: freestream velocity used [m/s]
    """
    y_span = []
    chord = []
    cl_vals = []
    in_section = False
    header_seen = False
    qinf_from_file = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Extract QInf from header if velocity not provided
    for line in lines[:20]:
        if line.startswith("QInf"):
            try:
                qinf_from_file = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass
            break

    # Use provided velocity or fall back to file value
    V = velocity if velocity is not None else qinf_from_file
    if V is None:
        raise ValueError("Could not determine velocity. Please specify --velocity.")

    # Parse Main Wing section
    for line in lines:
        s = line.strip()

        if s == "Main Wing":
            in_section = True
            header_seen = False
            continue

        if in_section and not header_seen:
            if s.startswith("y-span") or ("y-span" in s and "Cl" in s):
                header_seen = True
            continue

        if in_section and header_seen:
            if s == "":
                break

            cols = s.split()
            if len(cols) < 4:
                continue

            try:
                y = float(cols[0])
                c = float(cols[1])
                cl = float(cols[3])
            except (ValueError, IndexError):
                continue

            y_span.append(y)
            chord.append(c)
            cl_vals.append(cl)

    if not y_span:
        raise ValueError("Failed to parse Main Wing section.")

    # Compute lift per unit span: L'(y) = 0.5 * rho * V^2 * c(y) * Cl(y)
    q = 0.5 * air_density * V * V
    lift_per_span = [q * c * cl for c, cl in zip(chord, cl_vals)]

    return y_span, chord, cl_vals, lift_per_span, V


def plot_lift_distribution(y: List[float], lift: List[float], output_png: str, velocity: float):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Data points connected by line segments (piecewise linear)
    ax.plot(y, lift, marker="o", linewidth=1.5, markersize=4, color="tab:blue", label="Lift Distribution")
    ax.fill_between(y, lift, alpha=0.2, color="tab:blue")
    
    ax.set_title(f"Lift Force Distribution (Main Wing) - V = {velocity:.1f} m/s")
    ax.set_xlabel("y-span [m]")
    ax.set_ylabel("Lift per unit span [N/m]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()
    
    return fig


def write_csv(y: List[float], chord: List[float], cl: List[float], lift: List[float], output_csv: str, velocity: float, air_density: float):
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(f"# Lift Force Distribution (Piecewise Linear)\n")
        f.write(f"# Velocity [m/s]: {velocity}\n")
        f.write(f"# Air density [kg/m^3]: {air_density}\n")
        f.write(f"# Formula: L' [N/m] = 0.5 * rho * V^2 * c * Cl\n")
        f.write(f"# Interpolation: Use linear interpolation between data points\n")
        f.write("y_span [m],chord [m],Cl [-],lift_per_span [N/m]\n")
        for yi, ci, cli, li in zip(y, chord, cl, lift):
            f.write(f"{yi},{ci},{cli},{li}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot lift force distribution across the wing (XFLR5 export).")
    parser.add_argument("file_path", help="Path to XFLR5 text export containing 'Main Wing' section")
    parser.add_argument("--png", default="lift_distribution.png", help="Output PNG path")
    parser.add_argument("--csv", default="lift_distribution.csv", help="Output CSV path")
    parser.add_argument("--velocity", type=float, default=None, help="Override freestream velocity [m/s]")
    parser.add_argument("--air_density", type=float, default=1.225, help="Air density [kg/m^3] (default: 1.225)")
    args = parser.parse_args()

    y, chord, cl, lift, V = parse_main_wing_lift(
        args.file_path,
        air_density=args.air_density,
        velocity=args.velocity
    )
    
    plot_lift_distribution(y, lift, args.png, V)
    write_csv(y, chord, cl, lift, args.csv, V, args.air_density)

    # Calculate total lift by trapezoidal integration
    total_lift = 0.0
    for i in range(len(y) - 1):
        dy = abs(y[i+1] - y[i])
        avg_lift = (lift[i] + lift[i+1]) / 2
        total_lift += avg_lift * dy

    print(f"Saved plot to {args.png}")
    print(f"Saved data to {args.csv}")
    print(f"Velocity: {V:.2f} m/s")
    print(f"Span range: {min(y):.3f} to {max(y):.3f} m")
    print(f"Max lift per span: {max(lift):.2f} N/m (at y ≈ 0)")
    print(f"Estimated total lift (trapezoid): {total_lift:.2f} N")
    print(f"\nInterpolation: Piecewise linear (line segments between {len(y)} data points)")


if __name__ == "__main__":
    main()

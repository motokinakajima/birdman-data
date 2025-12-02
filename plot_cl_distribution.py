import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_main_wing_section(file_path: str) -> Tuple[List[float], List[float]]:
    y_span = []
    cl_vals = []
    in_section = False
    header_seen = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            # Enter section when we see exact "Main Wing"
            if s == "Main Wing":
                in_section = True
                header_seen = False
                continue

            if in_section and not header_seen:
                # Expect the header line with column names
                if s.startswith("y-span") or ("y-span" in s and "Cl" in s):
                    header_seen = True
                # skip until header line
                continue

            if in_section and header_seen:
                # Section ends on blank line
                if s == "":
                    break

                cols = s.split()
                # Expect at least: y-span, Chord, Ai, Cl, ...
                if len(cols) < 4:
                    continue

                try:
                    y = float(cols[0])
                    # Note: column 3 is capital "Cl" per file formatting
                    cl = float(cols[3])
                except Exception:
                    continue

                y_span.append(y)
                cl_vals.append(cl)

    if not y_span:
        raise ValueError("Failed to parse Main Wing section with y-span and Cl columns.")

    return y_span, cl_vals


def plot_cl(y: List[float], cl: List[float], output_png: str):
    plt.figure(figsize=(8, 4.5))
    plt.plot(y, cl, marker="o", linewidth=1.5)
    plt.title("Cl distribution across span (Main Wing)")
    plt.xlabel("y-span [m]")
    plt.ylabel("section Cl [-]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)


def write_csv(y: List[float], cl: List[float], output_csv: str):
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("y_span,Cl\n")
        for yi, cli in zip(y, cl):
            f.write(f"{yi},{cli}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot Cl distribution across the wing (XFLR5 export).")
    parser.add_argument("file_path", help="Path to XFLR5 text export containing 'Main Wing' section")
    parser.add_argument("--png", default="cl_distribution.png", help="Output PNG path")
    parser.add_argument("--csv", default="cl_distribution.csv", help="Output CSV path")
    args = parser.parse_args()

    y, cl = parse_main_wing_section(args.file_path)
    plot_cl(y, cl, args.png)
    write_csv(y, cl, args.csv)

    print(f"Saved plot to {args.png} and data to {args.csv}.")


if __name__ == "__main__":
    main()

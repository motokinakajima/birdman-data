import argparse
import csv
import datetime
import platform
import sys

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


if __name__ == "__main__":
    main()
"""
Compute lift force per rib by integrating lift distribution over each rib's domain.

For rib i at position x_i with neighbors x_{i-1} and x_{i+1}:
    Force_i = ∫_{(x_{i-1}+x_i)/2}^{(x_i+x_{i+1})/2} L'(x) dx

For endpoints, the integration limit is the span boundary instead of a midpoint.
"""

import argparse
import csv
import numpy as np
from typing import List, Tuple


def read_rib_positions(file_path: str) -> List[float]:
    """
    Read rib positions from CSV file (in mm).
    Skips comment lines starting with #.
    Returns positions in meters.
    """
    positions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()  # Remove comments
            if not line:
                continue
            try:
                pos_mm = float(line)
                positions.append(pos_mm / 1000.0)  # Convert mm to m
            except ValueError:
                continue
    return sorted(positions)


def read_lift_distribution(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read lift distribution CSV file.
    Returns (y_span, lift_per_span) arrays.
    """
    y_data = []
    lift_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "y_span" in line:  # Skip header
                continue
            cols = line.strip().split(",")
            if len(cols) >= 4:
                try:
                    y_data.append(float(cols[0]))
                    lift_data.append(float(cols[3]))
                except ValueError:
                    continue
    return np.array(y_data), np.array(lift_data)


def lift_interp(y_query: float, y_data: np.ndarray, lift_data: np.ndarray, 
                y_min: float, y_max: float, 
                wing_tip_left: float = None, wing_tip_right: float = None) -> float:
    """
    Interpolate lift at y_query.
    
    For regions between data boundary and actual wingtip, linearly extrapolate to 0.
    Returns 0 outside the actual wing span.
    """
    # Use data boundaries if wing tips not specified
    if wing_tip_left is None:
        wing_tip_left = y_min
    if wing_tip_right is None:
        wing_tip_right = y_max
    
    # Outside actual wing span
    if y_query < wing_tip_left or y_query > wing_tip_right:
        return 0.0
    
    # Inside data range: normal interpolation
    if y_min <= y_query <= y_max:
        return np.interp(y_query, y_data, lift_data)
    
    # Left extrapolation region: wing_tip_left < y_query < y_min
    if y_query < y_min:
        lift_at_boundary = lift_data[0]  # Lift at leftmost data point
        # Linear taper from boundary to 0 at wing tip
        fraction = (y_query - wing_tip_left) / (y_min - wing_tip_left)
        return lift_at_boundary * fraction
    
    # Right extrapolation region: y_max < y_query < wing_tip_right
    if y_query > y_max:
        lift_at_boundary = lift_data[-1]  # Lift at rightmost data point
        # Linear taper from boundary to 0 at wing tip
        fraction = (wing_tip_right - y_query) / (wing_tip_right - y_max)
        return lift_at_boundary * fraction
    
    return 0.0


def integrate_lift(y_start: float, y_end: float, y_data: np.ndarray, 
                   lift_data: np.ndarray, y_min: float, y_max: float,
                   wing_tip_left: float = None, wing_tip_right: float = None,
                   n_points: int = 50) -> float:
    """
    Numerically integrate lift distribution from y_start to y_end using trapezoidal rule.
    Clamps integration limits to actual wing tips.
    """
    if wing_tip_left is None:
        wing_tip_left = y_min
    if wing_tip_right is None:
        wing_tip_right = y_max
    
    # Clamp to actual wing span
    y_start_clamped = max(y_start, wing_tip_left)
    y_end_clamped = min(y_end, wing_tip_right)
    
    if y_start_clamped >= y_end_clamped:
        return 0.0
    
    # Create integration points
    y_interp = np.linspace(y_start_clamped, y_end_clamped, n_points)
    lift_interp_vals = np.array([
        lift_interp(y, y_data, lift_data, y_min, y_max, wing_tip_left, wing_tip_right) 
        for y in y_interp
    ])
    
    # Trapezoidal integration
    return np.trapz(lift_interp_vals, y_interp)


def compute_rib_forces(rib_positions: List[float], y_data: np.ndarray, 
                       lift_data: np.ndarray,
                       wing_tip_left: float = None, 
                       wing_tip_right: float = None) -> List[Tuple[float, float]]:
    """
    Compute lift force for each rib.
    
    Args:
        rib_positions: List of rib y-positions [m]
        y_data: Lift distribution y-positions from XFLR5
        lift_data: Lift per span values [N/m]
        wing_tip_left: Actual left wingtip position (default: min of rib positions)
        wing_tip_right: Actual right wingtip position (default: max of rib positions)
    
    Returns list of (rib_position, force) tuples.
    """
    y_min = y_data.min()
    y_max = y_data.max()
    
    # Default wing tips to rib extremes
    if wing_tip_left is None:
        wing_tip_left = min(rib_positions)
    if wing_tip_right is None:
        wing_tip_right = max(rib_positions)
    
    results = []
    n_ribs = len(rib_positions)
    
    for i, rib_pos in enumerate(rib_positions):
        # Determine integration limits
        if i == 0:
            # First rib: left limit is wing tip
            left_limit = wing_tip_left
        else:
            # Midpoint to previous rib
            left_limit = (rib_positions[i - 1] + rib_pos) / 2.0
        
        if i == n_ribs - 1:
            # Last rib: right limit is wing tip
            right_limit = wing_tip_right
        else:
            # Midpoint to next rib
            right_limit = (rib_pos + rib_positions[i + 1]) / 2.0
        
        # Integrate
        force = integrate_lift(left_limit, right_limit, y_data, lift_data, 
                               y_min, y_max, wing_tip_left, wing_tip_right)
        results.append((rib_pos, force))
    
    return results


def write_output(output_path: str, results: List[Tuple[float, float]], 
                 y_min: float, y_max: float,
                 wing_tip_left: float, wing_tip_right: float):
    """Write results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Lift Force per Rib\n")
        f.write(f"# Lift data span [m]: {y_min:.4f} to {y_max:.4f}\n")
        f.write(f"# Actual wing span [m]: {wing_tip_left:.4f} to {wing_tip_right:.4f}\n")
        f.write("# Force = integral of lift distribution over rib's domain\n")
        f.write("# Domain: midpoint to prev rib -> midpoint to next rib\n")
        f.write("# Endpoints use wing tip instead of midpoint\n")
        f.write("# Extrapolation: linear taper to 0 beyond lift data range\n")
        f.write("rib_position [m],force [N]\n")
        
        total_force = 0.0
        for rib_pos, force in results:
            f.write(f"{rib_pos},{force:.6f}\n")
            total_force += force
        
        f.write(f"# Total lift force: {total_force:.2f} N\n")


def main():
    parser = argparse.ArgumentParser(description="Compute lift force per rib")
    parser.add_argument("--ribs", required=True, help="Path to rib positions CSV (in mm)")
    parser.add_argument("--lift", required=True, help="Path to lift distribution CSV")
    parser.add_argument("--output", default="rib_forces.csv", help="Output CSV path")
    parser.add_argument("--wing_tip", type=float, default=None, 
                        help="Actual wing semi-span [m]. Wing tips at ±this value. "
                             "If not specified, uses max rib position.")
    args = parser.parse_args()
    
    # Read inputs
    rib_positions = read_rib_positions(args.ribs)
    y_data, lift_data = read_lift_distribution(args.lift)
    
    y_min, y_max = y_data.min(), y_data.max()
    
    # Determine actual wing tips
    if args.wing_tip is not None:
        wing_tip_left = -args.wing_tip
        wing_tip_right = args.wing_tip
    else:
        wing_tip_left = min(rib_positions)
        wing_tip_right = max(rib_positions)
    
    print(f"Loaded {len(rib_positions)} rib positions")
    print(f"Rib range: {min(rib_positions):.4f} to {max(rib_positions):.4f} m")
    print(f"Lift data range: {y_min:.4f} to {y_max:.4f} m")
    print(f"Actual wing tips: {wing_tip_left:.4f} to {wing_tip_right:.4f} m")
    
    if wing_tip_right > y_max:
        print(f"  -> Extrapolating lift linearly to 0 from {y_max:.4f} to {wing_tip_right:.4f} m")
    
    # Compute forces
    results = compute_rib_forces(rib_positions, y_data, lift_data, 
                                  wing_tip_left, wing_tip_right)
    
    # Write output
    write_output(args.output, results, y_min, y_max, wing_tip_left, wing_tip_right)
    
    total_force = sum(f for _, f in results)
    print(f"\nOutput: {args.output}")
    print(f"Total lift force: {total_force:.2f} N")
    print(f"Max force per rib: {max(f for _, f in results):.2f} N")


if __name__ == "__main__":
    main()

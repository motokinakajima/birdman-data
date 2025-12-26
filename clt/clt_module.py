"""
Modular Classical Laminating Theory (CLT) calculator with JSON I/O.

This module provides functions for CLT analysis of composite laminates:
- Material property management via JSON
- ABD matrix calculation
- Equivalent orthotropic and isotropic properties
- JSON/CSV output for results
"""

import numpy as np
import json
import csv
from typing import List, Tuple, Dict
from pathlib import Path


# =============================================================================
# Material Database Management
# =============================================================================

def load_materials(json_path: str = "materials.json") -> Dict[str, Dict]:
    """
    Load material properties from JSON file.
    
    Returns:
        dict: {material_name: {E1_MPa, E2_MPa, G12_MPa, nu12, Ez_MPa, nu23, thickness_mm}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        materials = json.load(f)
    return materials


def load_layup(json_path: str) -> Dict:
    """
    Load laminate layup definition from JSON file.
    
    Expected format:
    {
      "name": "Layup_Name",
      "description": "Description",
      "plies": [
        {"material": "Material_Name", "angle": 0, "thickness": null},
        ...
      ]
    }
    
    Returns:
        dict with name, description, and list of plies
    """
    with open(json_path, "r", encoding="utf-8") as f:
        layup = json.load(f)
    return layup


# =============================================================================
# CLT Core Functions
# =============================================================================

def build_Q(E1: float, E2: float, G12: float, nu12: float) -> np.ndarray:
    """
    Build the reduced stiffness matrix Q for a ply.
    
    Args:
        E1, E2: Young's moduli in fiber and transverse directions [MPa]
        G12: Shear modulus [MPa]
        nu12: Major Poisson's ratio
    
    Returns:
        Q: 3x3 reduced stiffness matrix [MPa]
    """
    nu21 = nu12 * E2 / E1
    denom = 1 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12
    Q = np.array([[Q11, Q12, 0.0],
                  [Q12, Q22, 0.0],
                  [0.0,  0.0,  Q66]])
    return Q


def Qbar_from_Q_and_theta(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Transform Q matrix to global coordinates using fiber angle.
    
    Args:
        Q: 3x3 reduced stiffness matrix [MPa]
        theta_deg: Fiber orientation angle [degrees]
    
    Returns:
        Qbar: 3x3 transformed stiffness matrix [MPa]
    """
    th = np.deg2rad(theta_deg)
    c = np.cos(th)
    s = np.sin(th)
    Q11, Q12, Q66 = Q[0,0], Q[0,1], Q[2,2]
    Q22 = Q[1,1]
    c2 = c*c
    s2 = s*s
    c3 = c2*c
    s3 = s2*s
    c4 = c2*c2
    s4 = s2*s2
    c2s2 = c2*s2

    Qbar11 = Q11*c4 + 2*(Q12 + 2*Q66)*c2s2 + Q22*s4
    Qbar12 = Q12*(c4 + s4) + (Q11 + Q22 - 4*Q66)*c2s2
    Qbar22 = Q11*s4 + 2*(Q12 + 2*Q66)*c2s2 + Q22*c4
    Qbar16 = (Q11 - Q12 - 2*Q66)*c3*s + (Q12 - Q22 + 2*Q66)*c*s3
    Qbar26 = (Q11 - Q12 - 2*Q66)*c*s3 + (Q12 - Q22 + 2*Q66)*c3*s
    Qbar66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*c2s2 + Q66*(c4 + s4)

    Qbar = np.array([[Qbar11, Qbar12, Qbar16],
                     [Qbar12, Qbar22, Qbar26],
                     [Qbar16, Qbar26, Qbar66]])
    return Qbar


def compute_z_from_thicknesses(thicknesses: List[float]) -> List[float]:
    """
    Compute z-coordinates from ply thicknesses.
    
    Args:
        thicknesses: List of ply thicknesses [mm]
    
    Returns:
        z: List of n+1 z-coordinates from bottom to top [mm]
    """
    t_total = sum(thicknesses)
    z = [-t_total/2.0]
    for ti in thicknesses:
        z.append(z[-1] + ti)
    return z


def compute_ABD(Qbar_list: List[np.ndarray], z_list: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ABD matrices for laminate.
    
    Args:
        Qbar_list: List of transformed stiffness matrices
        z_list: List of z-coordinates [mm]
    
    Returns:
        A: Extensional stiffness matrix [N/mm]
        B: Coupling stiffness matrix [N]
        D: Bending stiffness matrix [N·mm]
    """
    n = len(Qbar_list)
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    for k in range(n):
        zk = z_list[k+1]
        zkm = z_list[k]
        delta = zk - zkm
        A += Qbar_list[k] * delta
        B += 0.5 * Qbar_list[k] * (zk**2 - zkm**2)
        D += (1.0/3.0) * Qbar_list[k] * (zk**3 - zkm**3)
    return A, B, D


# =============================================================================
# Equivalent Properties
# =============================================================================

def equivalent_in_plane_from_A(A: np.ndarray, t_total: float) -> Tuple[float, float, float, float]:
    """
    Compute equivalent in-plane orthotropic properties from A matrix.
    
    Args:
        A: Extensional stiffness matrix [N/mm]
        t_total: Total laminate thickness [mm]
    
    Returns:
        Ex, Ey: In-plane moduli [MPa]
        Gxy: In-plane shear modulus [MPa]
        nu_xy: Major Poisson's ratio
    """
    Ainv = np.linalg.inv(A)
    Ex = 1.0 / (Ainv[0,0] * t_total)
    Ey = 1.0 / (Ainv[1,1] * t_total)
    nu_xy = -Ainv[0,1] * Ey * t_total
    Gxy = 1.0 / (Ainv[2,2] * t_total)
    return Ex, Ey, Gxy, nu_xy


def full_orthotropic_values(Ex: float, Ey: float, Gxy: float, nu_xy: float,
                            E2: float, G23: float, nu23: float) -> Dict[str, float]:
    """
    Compute full 9 orthotropic constants.
    
    Args:
        Ex, Ey, Gxy, nu_xy: From CLT A matrix
        E2, G23, nu23: From single lamina datasheet
    
    Returns:
        dict with Ex, Ey, Ez, nu_xy, nu_yz, nu_zx, Gxy, Gyz, Gzx [MPa]
    """
    Ez = E2
    Gyz = G23
    Gzx = G23
    nu_yz = nu23
    nu_zx = nu23

    return {
        "Ex": Ex,
        "Ey": Ey,
        "Ez": Ez,
        "nu_xy": nu_xy,
        "nu_yz": nu_yz,
        "nu_zx": nu_zx,
        "Gxy": Gxy,
        "Gyz": Gyz,
        "Gzx": Gzx
    }


def equivalent_isotropic_from_D(D: np.ndarray, t_total: float, 
                                nu: float = 0.28, mode: str = "avg") -> float:
    """
    Compute equivalent isotropic modulus for bending-only FEA.
    
    Args:
        D: Bending stiffness matrix [N·mm]
        t_total: Total thickness [mm]
        nu: Assumed isotropic Poisson's ratio
        mode: "D11", "D22", or "avg"
    
    Returns:
        E_eq: Equivalent isotropic modulus [MPa]
    """
    if mode == "D11":
        D_eq = D[0,0]
    elif mode == "D22":
        D_eq = D[1,1]
    elif mode == "avg":
        D_eq = 0.5 * (D[0,0] + D[1,1])
    else:
        raise ValueError("mode must be 'D11', 'D22', or 'avg'")

    E_eq = 12.0 * (1.0 - nu**2) * D_eq / (t_total**3)
    return E_eq


# =============================================================================
# Laminate Definition and Analysis
# =============================================================================

class Laminate:
    """Represents a composite laminate with CLT analysis."""
    
    def __init__(self, name: str = "Laminate"):
        self.name = name
        self.plies = []  # List of (material_name, angle, thickness_override)
        self.materials = None
        
    def add_ply(self, material_name: str, angle: float, thickness: float = None):
        """
        Add a ply to the laminate.
        
        Args:
            material_name: Name of material from materials database
            angle: Fiber orientation angle [degrees]
            thickness: Override thickness [mm] (None = use material default)
        """
        self.plies.append((material_name, angle, thickness))
    
    def analyze(self, materials_db: Dict[str, Dict]) -> Dict:
        """
        Perform CLT analysis on the laminate.
        
        Args:
            materials_db: Material properties dictionary
        
        Returns:
            dict with all analysis results
        """
        self.materials = materials_db
        
        # Build ply data
        thicknesses = []
        Qbar_list = []
        
        for mat_name, angle, t_override in self.plies:
            mat = materials_db[mat_name]
            thickness = t_override if t_override is not None else mat["thickness_mm"]
            thicknesses.append(thickness)
            
            Q = build_Q(mat["E1_MPa"], mat["E2_MPa"], mat["G12_MPa"], mat["nu12"])
            Qbar = Qbar_from_Q_and_theta(Q, angle)
            Qbar_list.append(Qbar)
        
        # Compute ABD matrices
        z = compute_z_from_thicknesses(thicknesses)
        A, B, D = compute_ABD(Qbar_list, z)
        t_total = sum(thicknesses)
        
        # Equivalent properties
        Ex, Ey, Gxy, nu_xy = equivalent_in_plane_from_A(A, t_total)
        
        # Use first ply for Ez, G23, nu23 (assuming similar materials)
        first_mat = materials_db[self.plies[0][0]]
        full9 = full_orthotropic_values(Ex, Ey, Gxy, nu_xy,
                                        first_mat["E2_MPa"], first_mat["G12_MPa"], first_mat["nu23"])
        
        E_iso = equivalent_isotropic_from_D(D, t_total, nu=0.28, mode="avg")
        
        return {
            "name": self.name,
            "t_total": t_total,
            "n_plies": len(self.plies),
            "plies": self.plies,
            "A": A,
            "B": B,
            "D": D,
            "Ex": Ex,
            "Ey": Ey,
            "Gxy": Gxy,
            "nu_xy": nu_xy,
            "full_orthotropic": full9,
            "E_iso": E_iso,
            "nu_iso": 0.28
        }
    
    @classmethod
    def from_json(cls, json_path: str):
        """
        Create a Laminate from a JSON file.
        
        Args:
            json_path: Path to JSON layup definition file
        
        Returns:
            Laminate object
        """
        layup_data = load_layup(json_path)
        lam = cls(name=layup_data.get("name", "Laminate"))
        
        for ply in layup_data["plies"]:
            lam.add_ply(
                ply["material"],
                ply["angle"],
                ply.get("thickness")  # None if not specified
            )
        
        return lam


# =============================================================================
# Output Functions
# =============================================================================

def write_results_to_json(results: Dict, output_path: str):
    """Write CLT analysis results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    output = {
        "name": results["name"],
        "t_total_mm": results["t_total"],
        "n_plies": results["n_plies"],
        "plies": [{"material": p[0], "angle": p[1], "thickness": p[2]} for p in results["plies"]],
        "A_matrix_N_per_mm": results["A"].tolist(),
        "B_matrix_N": results["B"].tolist(),
        "D_matrix_N_mm": results["D"].tolist(),
        "in_plane_properties": {
            "Ex_MPa": results["Ex"],
            "Ey_MPa": results["Ey"],
            "Gxy_MPa": results["Gxy"],
            "nu_xy": results["nu_xy"]
        },
        "full_orthotropic": results["full_orthotropic"],
        "equivalent_isotropic": {
            "E_iso_MPa": results["E_iso"],
            "nu_iso": results["nu_iso"]
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def write_results_to_csv(results: Dict, output_path: str):
    """Write CLT analysis results to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# CLT Analysis Results: {results['name']}\n")
        f.write(f"# Total thickness: {results['t_total']:.6f} mm\n")
        f.write(f"# Number of plies: {results['n_plies']}\n")
        f.write("#\n")
        
        writer = csv.writer(f)
        
        # In-plane properties
        writer.writerow(["Property", "Value", "Unit"])
        writer.writerow(["Ex", f"{results['Ex']:.3f}", "MPa"])
        writer.writerow(["Ey", f"{results['Ey']:.3f}", "MPa"])
        writer.writerow(["Gxy", f"{results['Gxy']:.3f}", "MPa"])
        writer.writerow(["nu_xy", f"{results['nu_xy']:.4f}", "-"])
        writer.writerow([])
        
        # Full orthotropic
        writer.writerow(["Full Orthotropic Properties", "", ""])
        orth = results["full_orthotropic"]
        for key in ["Ex", "Ey", "Ez", "Gxy", "Gyz", "Gzx"]:
            writer.writerow([key, f"{orth[key]:.3f}", "MPa"])
        for key in ["nu_xy", "nu_yz", "nu_zx"]:
            writer.writerow([key, f"{orth[key]:.4f}", "-"])
        writer.writerow([])
        
        # Isotropic equivalent
        writer.writerow(["Equivalent Isotropic (for FEA)", "", ""])
        writer.writerow(["E_iso", f"{results['E_iso']:.1f}", "MPa"])
        writer.writerow(["nu_iso", f"{results['nu_iso']:.2f}", "-"])


def print_results(results: Dict):
    """Print CLT analysis results to console."""
    print(f"\n{'='*60}")
    print(f"CLT Analysis: {results['name']}")
    print(f"{'='*60}")
    print(f"Total thickness: {results['t_total']:.6f} mm")
    print(f"Number of plies: {results['n_plies']}")
    
    print(f"\n--- ABD Matrices ---")
    np.set_printoptions(precision=3, suppress=True)
    print(f"A [N/mm]:\n{results['A']}")
    print(f"\nB [N]:\n{results['B']}")
    print(f"\nD [N·mm]:\n{results['D']}")
    
    print(f"\n--- Equivalent In-Plane Properties (from A) ---")
    print(f"Ex    = {results['Ex']:.3f} MPa")
    print(f"Ey    = {results['Ey']:.3f} MPa")
    print(f"Gxy   = {results['Gxy']:.3f} MPa")
    print(f"nu_xy = {results['nu_xy']:.4f}")
    
    print(f"\n--- Full Orthotropic Constants ---")
    orth = results["full_orthotropic"]
    print(f"Ex    = {orth['Ex']:.3f} MPa")
    print(f"Ey    = {orth['Ey']:.3f} MPa")
    print(f"Ez    = {orth['Ez']:.3f} MPa")
    print(f"Gxy   = {orth['Gxy']:.3f} MPa")
    print(f"Gyz   = {orth['Gyz']:.3f} MPa")
    print(f"Gzx   = {orth['Gzx']:.3f} MPa")
    print(f"ν_xy  = {orth['nu_xy']:.4f}")
    print(f"ν_yz  = {orth['nu_yz']:.4f}")
    print(f"ν_zx  = {orth['nu_zx']:.4f}")
    
    print(f"\n--- Equivalent Isotropic (for Bending FEA) ---")
    print(f"E_iso  = {results['E_iso']:.1f} MPa")
    print(f"nu_iso = {results['nu_iso']:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import glob
    import os
    
    # Load materials database
    materials = load_materials("materials.json")
    
    # Find all layup files in plys/ directory
    layup_files = glob.glob("plys/*.json")
    
    if not layup_files:
        print("No layup files found in plys/ directory")
        print("Please add layup JSON files to the plys/ folder")
        exit(1)
    
    print(f"Found {len(layup_files)} layup file(s) in plys/")
    print("=" * 70)
    
    # Process each layup file
    for layup_path in layup_files:
        layup_name = os.path.basename(layup_path).replace(".json", "")
        print(f"\nProcessing: {layup_name}")
        print("-" * 70)
        
        try:
            # Load and analyze
            lam = Laminate.from_json(layup_path)
            results = lam.analyze(materials)
            
            # Print results
            print_results(results)
            
            # Save to results/ directory
            csv_output = f"results/{layup_name}.csv"
            json_output = f"results/{layup_name}.json"
            
            write_results_to_csv(results, csv_output)
            write_results_to_json(results, json_output)
            
            print(f"✓ Results saved:")
            print(f"  - {csv_output}")
            print(f"  - {json_output}")
            
        except Exception as e:
            print(f"✗ Error processing {layup_name}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"Batch processing complete: {len(layup_files)} layup(s) processed")
    print("Results saved to results/ directory")

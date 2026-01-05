"""
Plot z-displacement (wing deflection) along the span.
"""

import matplotlib.pyplot as plt
import numpy as np

# Read data
positions = []
displacements = []

with open("z-displacement.csv", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        # Remove comments and parse
        line = line.split("#")[0].strip()
        if not line or "?" in line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                pos = float(parts[0].strip())
                disp = float(parts[1].strip())
                positions.append(pos)
                displacements.append(disp)
            except ValueError:
                continue

# Convert to numpy arrays
positions = np.array(positions)
displacements = np.array(displacements)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(positions, displacements, marker="o", linewidth=2, markersize=6, color="tab:blue")
plt.fill_between(positions, displacements, alpha=0.2, color="tab:blue")

plt.xlabel("Keta Position [mm]")
plt.ylabel("Z Displacement [mm]")
plt.title("Wing Z-Displacement (Deflection) Distribution")
plt.grid(True, linestyle=":", alpha=0.6)
plt.axhline(y=0, color="k", linewidth=0.5)

# Annotate dihedral start
dihedral_start = 4750
plt.axvline(x=dihedral_start, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Dihedral start (4750mm)")
plt.legend()

plt.tight_layout()
plt.savefig("z-displacement_plot.png", dpi=150)
print(f"Plot saved: z-displacement_plot.png")
print(f"Data points: {len(positions)}")
print(f"Max displacement: {max(displacements):.2f} mm at position {positions[np.argmax(displacements)]:.0f} mm")

plt.show()

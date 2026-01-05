"""
Calculate and plot the bending angle at each point (difference between left and right tangent angles).
各点での曲がり角度を計算（左からの接線と右への接線の角度差）
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

positions = np.array(positions)
displacements = np.array(displacements)

# Calculate bending angle at each point
# 各点での左からの接線と右への接線の角度差を計算
bending_angles = []
point_positions = []

for i in range(len(positions) - 1):  # 最後の点は除外
    x_current = positions[i]
    z_current = displacements[i]
    
    # 左からの接線の傾き（最初の点は0として扱う）
    if i == 0:
        # 最初の点：左からの傾きは0度
        angle_left = 0.0
    else:
        x_prev = positions[i-1]
        z_prev = displacements[i-1]
        angle_left = np.degrees(np.arctan((z_current - z_prev) / (x_current - x_prev)))
    
    # 右への接線の傾き
    x_next = positions[i+1]
    z_next = displacements[i+1]
    angle_right = np.degrees(np.arctan((z_next - z_current) / (x_next - x_current)))
    
    # 角度差（曲がり角度）
    bending_angle = angle_right - angle_left
    
    bending_angles.append(bending_angle)
    point_positions.append(x_current)
    
    print(f"Point {i+1} at x={x_current:.0f} mm:")
    print(f"  Left tangent angle: {angle_left:.3f}°")
    print(f"  Right tangent angle: {angle_right:.3f}°")
    print(f"  Bending angle: {bending_angle:.3f}°")
    print()

bending_angles = np.array(bending_angles)
point_positions = np.array(point_positions)

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot z-displacement curve
ax.plot(positions, displacements, marker="o", linewidth=2, markersize=8, 
        color="tab:blue", label="Z-displacement", zorder=3)

# Add bending angle annotations next to each point (except the last one)
for i, (x, angle) in enumerate(zip(point_positions, bending_angles)):
    # Position text to the right of the point
    offset_x = 200  # mm
    offset_y = 20   # mm
    
    color = "red" if abs(angle) > 2 else "green"
    ax.annotate(f'{angle:.2f}°', 
                xy=(x, displacements[i]), 
                xytext=(x + offset_x, displacements[i] + offset_y),
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=5)

ax.set_xlabel("Keta Position [mm]", fontsize=12)
ax.set_ylabel("Z Displacement [mm]", fontsize=12)
ax.set_title("Wing Z-Displacement with Bending Angles", fontsize=14)
ax.grid(True, linestyle=":", alpha=0.6)
ax.axvline(x=4750, color="purple", linestyle="--", linewidth=2, alpha=0.7, label="Dihedral start (4750mm)")
ax.legend(fontsize=10)
ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.savefig("virtual_dihedral_angle.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved: virtual_dihedral_angle.png")
print(f"\nTotal bending: {np.sum(bending_angles):.3f}°")

plt.show()

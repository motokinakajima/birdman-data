"""
Calculate Clβ (rolling moment derivative with respect to sideslip angle)
using linear least squares regression near β=0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read data
data = np.loadtxt("Beta_Cl_data.csv", delimiter=",", skiprows=1)
beta_all_deg = data[:, 0]
cl_all = data[:, 1]

# Convert to radians
beta_all_rad = np.radians(beta_all_deg)

# Define range for linear fit
beta_range_deg = 1.0  # ±1° (change to 2.0 for ±2°)
beta_range_rad = np.radians(beta_range_deg)

# Filter data near β=0
mask = np.abs(beta_all_deg) <= beta_range_deg
beta_fit_deg = beta_all_deg[mask]
beta_fit_rad = beta_all_rad[mask]
cl_fit = cl_all[mask]

print(f"Linear regression range: β = ±{beta_range_deg}° (±{beta_range_rad:.6f} rad)")
print(f"Number of points used: {len(beta_fit_rad)}")
print(f"Beta range: {beta_fit_deg.min():.2f}° to {beta_fit_deg.max():.2f}° ({beta_fit_rad.min():.6f} to {beta_fit_rad.max():.6f} rad)")
print()

# Perform linear regression: Cl = a*beta + b (beta in radians)
slope, intercept, r_value, p_value, std_err = stats.linregress(beta_fit_rad, cl_fit)

print("="*60)
print("LINEAR REGRESSION RESULTS")
print("="*60)
print(f"Cl(β) ≈ {slope:.6f} * β + {intercept:.6f}  (β in radians)")
print()
print(f"Clβ (slope)        : {slope:.6f} [1/rad]")
print(f"Intercept (Cl₀)    : {intercept:.6f}")
print(f"R² (linearity)     : {r_value**2:.6f}")
print(f"Standard error     : {std_err:.6e}")
print(f"P-value            : {p_value:.6e}")
print("="*60)
print()

# Additional analysis
residuals = cl_fit - (slope * beta_fit_rad + intercept)
rmse = np.sqrt(np.mean(residuals**2))
max_residual = np.max(np.abs(residuals))

print("RESIDUAL ANALYSIS")
print(f"RMSE               : {rmse:.6e}")
print(f"Max residual       : {max_residual:.6e}")
print()

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: All data with linear fit (using degrees for x-axis display)
ax1.plot(beta_all_deg, cl_all, 'o', color='lightblue', markersize=4, label='All data', alpha=0.5)
ax1.plot(beta_fit_deg, cl_fit, 'o', color='blue', markersize=6, label=f'Fit region (±{beta_range_deg}°)')

# Plot regression line
beta_line_rad = np.linspace(beta_fit_rad.min(), beta_fit_rad.max(), 100)
beta_line_deg = np.degrees(beta_line_rad)
cl_line = slope * beta_line_rad + intercept
ax1.plot(beta_line_deg, cl_line, 'r-', linewidth=2, label=f'Linear fit: Cl = {slope:.6f}β + {intercept:.6f} (β in rad)')

ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.set_xlabel('β (sideslip angle) [deg]', fontsize=11)
ax1.set_ylabel('Cl (rolling moment coefficient)', fontsize=11)
ax1.set_title(f'Cl vs β with Linear Fit (R² = {r_value**2:.6f})', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)

# Plot 2: Residuals
ax2.plot(beta_fit_deg, residuals * 1e6, 'o', color='green', markersize=6)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
ax2.fill_between(beta_fit_deg, -rmse*1e6, rmse*1e6, alpha=0.2, color='yellow', label=f'±RMSE = ±{rmse:.2e}')
ax2.set_xlabel('β (sideslip angle) [deg]', fontsize=11)
ax2.set_ylabel('Residuals [×10⁻⁶]', fontsize=11)
ax2.set_title('Residual Analysis', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("Clbeta_analysis.png", dpi=150)
print(f"Plot saved: Clbeta_analysis.png")

# Save results to file
with open("Clbeta_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Clβ CALCULATION RESULTS\n")
    f.write(f"="*60 + "\n")
    f.write(f"Data file          : Beta_Cl_data.csv\n")
    f.write(f"Regression range   : β = ±{beta_range_deg}° (±{beta_range_rad:.6f} rad)\n")
    f.write(f"Number of points   : {len(beta_fit_rad)}\n")
    f.write(f"\n")
    f.write(f"LINEAR FIT: Cl(β) ≈ {slope:.6f} * β + {intercept:.6f}  (β in radians)\n")
    f.write(f"\n")
    f.write(f"Clβ (derivative)   : {slope:.6f} [1/rad]\n")
    f.write(f"Intercept (Cl₀)    : {intercept:.6f}\n")
    f.write(f"R² (linearity)     : {r_value**2:.6f}\n")
    f.write(f"Standard error     : {std_err:.6e}\n")
    f.write(f"RMSE               : {rmse:.6e}\n")
    f.write(f"Max residual       : {max_residual:.6e}\n")
    f.write(f"="*60 + "\n")

print(f"Results saved: Clbeta_result.txt")
plt.show()

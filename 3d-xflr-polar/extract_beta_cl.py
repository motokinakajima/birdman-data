"""
Extract Beta and Cl from XFLR5 VLM2 output file.
"""

import csv

input_file = "T5-a4_5°-8_5m_s-VLM2.txt"
output_file = "Beta_Cl_data.csv"

beta_values = []
cl_values = []

# Read data from the file
with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        
        # Skip empty lines and header lines
        if not line or line.startswith("xflr5") or line.startswith("Plane") or line.startswith("Polar") or line.startswith("alpha"):
            continue
        
        # Parse data lines
        parts = line.split()
        if len(parts) >= 8:
            try:
                # Column indices: alpha=0, Beta=1, CL=2, ..., Cl=7
                beta = float(parts[1])
                cl = float(parts[7])
                
                beta_values.append(beta)
                cl_values.append(cl)
            except (ValueError, IndexError):
                continue

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Beta", "Cl"])
    
    for beta, cl in zip(beta_values, cl_values):
        writer.writerow([beta, cl])

print(f"Data extracted: {len(beta_values)} points")
print(f"Beta range: {min(beta_values):.1f} to {max(beta_values):.1f} degrees")
print(f"Cl range: {min(cl_values):.6f} to {max(cl_values):.6f}")
print(f"Output saved to: {output_file}")

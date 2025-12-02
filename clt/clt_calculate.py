# ---------------------------------------------------------
# Improved CLT code that also outputs full 9 orthotropic values
# (Ex, Ey, Ez, nu_xy, nu_yz, nu_zx, Gxy, Gyz, Gzx)
# ---------------------------------------------------------
import numpy as np

def build_Q(E1, E2, G12, nu12):
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

def Qbar_from_Q_and_theta(Q, theta_deg):
    th = np.deg2rad(theta_deg)
    c = np.cos(th); s = np.sin(th)
    Q11, Q12, Q66 = Q[0,0], Q[0,1], Q[2,2]
    Q22 = Q[1,1]
    c2 = c*c; s2 = s*s
    c3 = c2*c; s3 = s2*s
    c4 = c2*c2; s4 = s2*s2
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

def compute_z_from_thicknesses(thicknesses):
    t_total = sum(thicknesses)
    z = [-t_total/2.0]
    for ti in thicknesses:
        z.append(z[-1] + ti)
    return z

def compute_ABD(Qbar_list, z_list):
    n = len(Qbar_list)
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    for k in range(n):
        zk = z_list[k+1]; zkm = z_list[k]
        delta = zk - zkm
        A += Qbar_list[k] * delta
        B += 0.5 * Qbar_list[k] * (zk**2 - zkm**2)
        D += (1.0/3.0) * Qbar_list[k] * (zk**3 - zkm**3)
    return A, B, D

def equivalent_in_plane_from_A(A, t_total):
    Ainv = np.linalg.inv(A)
    Ex = 1.0 / (Ainv[0,0] * t_total)
    Ey = 1.0 / (Ainv[1,1] * t_total)
    nu_xy = -Ainv[0,1] * Ey * t_total
    Gxy = 1.0 / (Ainv[2,2] * t_total)
    return Ex, Ey, Gxy, nu_xy

# ----------------------------------------------------------------
# NEW: compute full 9 orthotropic values
# ----------------------------------------------------------------
def full_orthotropic_values(Ex, Ey, Gxy, nu_xy, 
                            E2, G23, nu23):
    """
    Ex, Ey, Gxy, nu_xy -> from CLT
    E2, G23, nu23      -> from single lamina datasheet

    Returns:
        Ex, Ey, Ez,
        nu_xy, nu_yz, nu_zx,
        Gxy, Gyz, Gzx
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

# -----------------------------------------------------------
# Demo (same ply stack as before) — REPLACE material data!
# -----------------------------------------------------------

plys = [
    # (MaterialName, FiberAngle[deg], Thickness[mm],
    #  [E1[MPa], E2[MPa], G12[MPa], nu12, Ez[MPa], nu23])
    ("P3252S-12", 90,   0.125, [30000.0, 10000.0, 4000.0, 0.3,  3000.0, 0.45]),
    ("HRX350G125S", +45, 0.111, [135000.0, 9000.0, 5000.0, 0.3, 3000.0, 0.45]),
    ("HRX350G125S", 0,    0.111, [135000.0, 9000.0, 5000.0, 0.3, 3000.0, 0.45]),
    ("HRX350G125S", -45, 0.111, [135000.0, 9000.0, 5000.0, 0.3, 3000.0, 0.45]),
    ("HRX350G125S", 0,    0.111, [135000.0, 9000.0, 5000.0, 0.3, 3000.0, 0.45]),
    ("HRX350G125S", +45, 0.111, [135000.0, 9000.0, 5000.0, 0.3, 3000.0, 0.45]),
    ("P3252S-12", 90,   0.125, [30000.0, 10000.0, 4000.0, 0.3, 3000.0, 0.45])
]

# thickness
thicknesses = [p[2] for p in plys]
z = compute_z_from_thicknesses(thicknesses)

Qbar_list = []
for name, angle, ti, props in plys:
    E1, E2, G12, nu12, G23, nu23 = props
    Q = build_Q(E1, E2, G12, nu12)
    Qbar_list.append(Qbar_from_Q_and_theta(Q, angle))

A, B, D = compute_ABD(Qbar_list, z)
t_total = sum(thicknesses)

Ex, Ey, Gxy, nu_xy = equivalent_in_plane_from_A(A, t_total)

# pick one lamina (assume similar) to fetch Ez,G23,nu23
_, _, _, [E1, E2, G12, nu12, G23, nu23] = plys[1]

full9 = full_orthotropic_values(Ex, Ey, Gxy, nu_xy, E2, G23, nu23)

print("Total thickness t = {:.6f} mm".format(t_total))

np.set_printoptions(precision=3, suppress=True)

print("\nA matrix [N/mm]:\n", A)
print("\nB matrix [N]:\n", B)
print("\nD matrix [N·mm]:\n", D)

# Equivalent in-plane (orthotropic) properties from A
Ex, Ey, Gxy, nu_xy = equivalent_in_plane_from_A(A, t_total)
print("\nEquivalent in-plane orthotropic properties (from A):")
print("Ex = {:.3f} MPa".format(Ex))
print("Ey = {:.3f} MPa".format(Ey))
print("Gxy = {:.3f} MPa".format(Gxy))
print("nu_xy = {:.4f}".format(nu_xy))

print("\nFull 9 orthotropic constants (unit included):")
print("Ex   = {:.3f} MPa".format(full9["Ex"]))
print("Ey   = {:.3f} MPa".format(full9["Ey"]))
print("Ez   = {:.3f} MPa".format(full9["Ez"]))

print("Gxy  = {:.3f} MPa".format(full9["Gxy"]))
print("Gyz  = {:.3f} MPa".format(full9["Gyz"]))
print("Gzx  = {:.3f} MPa".format(full9["Gzx"]))

print("ν_xy = {:.4f}".format(full9["nu_xy"]))
print("ν_yz = {:.4f}".format(full9["nu_yz"]))
print("ν_zx = {:.4f}".format(full9["nu_zx"]))


print("\n--- Unit Notes ---")
print(" - All elastic moduli: MPa")
print(" - Thickness: mm")
print(" - A matrix: N/mm")
print(" - B matrix: N")
print(" - D matrix: N·mm")
print(" - Poisson ratios are dimensionless")


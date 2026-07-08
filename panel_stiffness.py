"""
panel_stiffness.py

Predicts the out-of-plane (central load / central deflection) stiffness of a
simply-supported composite laminate panel under a hemispherical indenter,
and compares it against experimental data.

Pipeline
--------
1. Classical Lamination Theory (CLT)      -> full A, B, D matrices per laminate
2. First-order Shear Deformation Theory   -> Navier series solution for a
   (Reissner-Mindlin plate)                 simply supported rectangular plate,
                                             loaded over a small square patch
                                             (NOT a mathematical point load,
                                             which is singular in FSDT)
3. Hertz contact theory                   -> local indentation stiffness of the
                                             steel hemisphere against the
                                             composite surface
4. Series combination                     -> 1/k_total = 1/k_plate + 1/k_contact

Notes / assumptions (edit these for your own laminates)
---------------------------------------------------------
- Plate D16/D26 (bending-twist coupling) and any B matrix (bending-stretching
  coupling, present for non-symmetric layups e.g. literal DD stacking) are
  NOT included in the FSDT solve below -- that solution strictly requires a
  "specially orthotropic" plate (D16=D26=0, B=0). This is a good engineering
  approximation here (D16, D26 are 3-20% of D11 for these laminates) but if
  you need the coupling terms explicitly, the ABD() function already returns
  them -- the Navier solve would need generalising (more series unknowns).
- Transverse shear stiffness A44=A55 assumes the ply is transversely
  isotropic in shear (G13=G23), which holds for the material data used here.
  If your ply doesn't satisfy G13=G23, transverse shear stiffness becomes
  orientation-dependent and A44 != A55 in general.
- Hertz contact uses the ply's transverse modulus E22 (=E33) as a stand-in
  for the anisotropic composite surface modulus. This is an approximation;
  Hertz theory is strictly derived for isotropic half-spaces.
- Simply-supported span is taken as the clamped window, not the full panel
  (this matched the experimental data much better in earlier fitting -- see
  the sensitivity check in `demo_span_sensitivity()`).
"""

import numpy as np


# =====================================================================
# 1. Classical Lamination Theory
# =====================================================================

def ply_Q_matrix(E11, E22, nu12, G12):
    """Reduced stiffness matrix Q (3x3) for a UD ply, in material axes."""
    nu21 = nu12 * E22 / E11
    denom = 1 - nu12 * nu21
    Q11 = E11 / denom
    Q22 = E22 / denom
    Q12 = nu12 * Q22
    Q66 = G12
    return np.array([[Q11, Q12, 0.0],
                      [Q12, Q22, 0.0],
                      [0.0, 0.0, Q66]])


def rotate_Q(Q, theta_deg):
    """Transform ply stiffness matrix Q to laminate axes at angle theta_deg."""
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)

    Qbar11 = Q11 * m**4 + 2 * (Q12 + 2 * Q66) * m**2 * n**2 + Q22 * n**4
    Qbar22 = Q11 * n**4 + 2 * (Q12 + 2 * Q66) * m**2 * n**2 + Q22 * m**4
    Qbar12 = (Q11 + Q22 - 4 * Q66) * m**2 * n**2 + Q12 * (m**4 + n**4)
    Qbar16 = (Q11 - Q12 - 2 * Q66) * m**3 * n + (Q12 - Q22 + 2 * Q66) * m * n**3
    Qbar26 = (Q11 - Q12 - 2 * Q66) * m * n**3 + (Q12 - Q22 + 2 * Q66) * m**3 * n
    Qbar66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m**2 * n**2 + Q66 * (m**4 + n**4)

    return np.array([[Qbar11, Qbar12, Qbar16],
                      [Qbar12, Qbar22, Qbar26],
                      [Qbar16, Qbar26, Qbar66]])


def ABD_matrices(Q, layup, t_ply):
    """
    Full laminate A, B, D matrices.

    Q       : 3x3 ply stiffness matrix in material axes (from ply_Q_matrix)
    layup   : list of ply angles in degrees, top ply first
    t_ply   : single ply thickness (consistent length units, e.g. mm)

    Units: if Q is in GPa and t_ply in mm, then
        A is in GPa*mm = kN/mm   (in-plane stiffness per unit width)
        B is in GPa*mm^2         (coupling)
        D is in GPa*mm^3 = kN*mm (bending stiffness per unit width)
    """
    n = len(layup)
    h = n * t_ply
    z = [h / 2 - i * t_ply for i in range(n + 1)]  # z[0] = +h/2 (top) ... z[n] = -h/2

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, theta in enumerate(layup):
        Qb = rotate_Q(Q, theta)
        zk, zk1 = z[k], z[k + 1]
        A += Qb * (zk - zk1)
        B += 0.5 * Qb * (zk**2 - zk1**2)
        D += (1 / 3) * Qb * (zk**3 - zk1**3)
    return A, B, D


# =====================================================================
# 2. Shear-deformable (Reissner-Mindlin) Navier plate solution
#    Simply supported rectangular plate, patch load at plate centre
# =====================================================================

def transverse_shear_stiffness(G13, G23, h, k_shear=5 / 6):
    """
    Laminate transverse shear stiffness A44, A55 (through-thickness).
    Valid as written only if the ply is transversely isotropic in shear
    (G13 == G23), so that rotation doesn't change it -- true for the
    material data used in this analysis. For a general ply you would need
    to rotate G13/G23 per ply angle and integrate through the thickness.
    """
    assert abs(G13 - G23) < 1e-9, (
        "G13 != G23: transverse shear stiffness is orientation-dependent, "
        "extend this function to rotate per ply before using."
    )
    A44 = k_shear * G23 * h
    A55 = k_shear * G13 * h
    return A44, A55


def patch_load_fourier_coeff(P, m, n, a, b, c, x0=None, y0=None):
    """
    Fourier sine-series coefficient for a uniform pressure load of total
    force P, spread over a small SQUARE patch of half-width c, centred at
    (x0, y0) (default: plate centre). c -> 0 recovers a point load.
    """
    if x0 is None:
        x0 = a / 2
    if y0 is None:
        y0 = b / 2
    if c <= 0:
        return (4 * P / (a * b)) * np.sin(m * np.pi * x0 / a) * np.sin(n * np.pi * y0 / b)
    return (
        (4 * P) / (m * n * np.pi**2 * c**2)
        * np.sin(m * np.pi * x0 / a) * np.sin(m * np.pi * c / a)
        * np.sin(n * np.pi * y0 / b) * np.sin(n * np.pi * c / b)
    )


def fsdt_center_deflection(D, A44, A55, a, b, P, c, M=241, N=241):
    """
    Central deflection of a simply-supported, specially-orthotropic
    (D16=D26=0 assumed), shear-deformable rectangular plate under a
    patch load of total force P at the plate centre.

    D           : 3x3 bending stiffness matrix (only D11, D12, D22, D66 used)
    A44, A55    : transverse shear stiffnesses
    a, b        : plate span in x, y (e.g. the clamped window dimensions)
    P           : total applied force
    c           : patch half-width (use a physically meaningful value,
                  e.g. the Hertz contact radius -- NOT c=0, which is a
                  singular point load that will not converge)
    M, N        : number of odd series terms (converges quickly once c > 0)

    Returns central deflection w_center (same length units as a, b, and
    consistent with the units of D/A44/A55/P).
    """
    D11, D22, D66, D12 = D[0, 0], D[1, 1], D[2, 2], D[0, 1]
    x0, y0 = a / 2, b / 2
    w = 0.0
    for m in range(1, M + 1, 2):
        alpha = m * np.pi / a
        sx = np.sin(m * np.pi * x0 / a)   # response evaluated at load point (centre)
        for n in range(1, N + 1, 2):
            beta = n * np.pi / b
            sy = np.sin(n * np.pi * y0 / b)
            Qmn = patch_load_fourier_coeff(P, m, n, a, b, c, x0, y0)
            K = np.array([
                [A55 * alpha**2 + A44 * beta**2, A55 * alpha, A44 * beta],
                [A55 * alpha, D11 * alpha**2 + D66 * beta**2 + A55, (D12 + D66) * alpha * beta],
                [A44 * beta, (D12 + D66) * alpha * beta, D66 * alpha**2 + D22 * beta**2 + A44],
            ])
            Wmn = np.linalg.solve(K, np.array([Qmn, 0.0, 0.0]))[0]
            # response point (x0,y0) == load point here, so multiply by sin(m pi x0/a)*sin(n pi y0/b) again
            w += Wmn * sx * sy
    return w


def fsdt_deflection_field(D, A44, A55, a, b, P, c, x, y, M=121, N=81):
    """
    Full deflection field w(x,y) for the same simply-supported, patch-loaded
    plate as fsdt_center_deflection, evaluated over an arbitrary grid.

    x, y : 1D arrays of coordinates (mm), 0 <= x <= a, 0 <= y <= b
    Returns a (len(x), len(y)) array: field[i, j] = w(x[i], y[j])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D11, D22, D66, D12 = D[0, 0], D[1, 1], D[2, 2], D[0, 1]
    x0, y0 = a / 2, b / 2
    field = np.zeros((len(x), len(y)))
    for m in range(1, M + 1, 2):
        alpha = m * np.pi / a
        sx = np.sin(m * np.pi * x / a)
        for n in range(1, N + 1, 2):
            beta = n * np.pi / b
            sy = np.sin(n * np.pi * y / b)
            Qmn = patch_load_fourier_coeff(P, m, n, a, b, c, x0, y0)
            K = np.array([
                [A55 * alpha**2 + A44 * beta**2, A55 * alpha, A44 * beta],
                [A55 * alpha, D11 * alpha**2 + D66 * beta**2 + A55, (D12 + D66) * alpha * beta],
                [A44 * beta, (D12 + D66) * alpha * beta, D66 * alpha**2 + D22 * beta**2 + A44],
            ])
            Wmn = np.linalg.solve(K, np.array([Qmn, 0.0, 0.0]))[0]
            field += Wmn * np.outer(sx, sy)
    return field


def plate_stiffness(D, A44, A55, a, b, c, P_ref=1.0, M=241, N=241):
    """Plate-only central stiffness (force / deflection) at reference load P_ref."""
    w = fsdt_center_deflection(D, A44, A55, a, b, P_ref, c, M, N)
    return P_ref / w


# =====================================================================
# 3. Hertz contact (rigid sphere on a flat surface)
# =====================================================================

def hertz_contact(F, R, E1, nu1, E2, nu2):
    """
    Hertz elastic contact of a rigid-ish sphere (radius R) on a flat
    surface. Returns (contact radius, indentation depth, local contact
    stiffness dF/ddelta) at force F.

    E1, nu1 : indenter modulus/Poisson ratio
    E2, nu2 : substrate modulus/Poisson ratio (an isotropic approximation
              for the composite surface -- see module docstring)
    Units: use consistent units throughout, e.g. GPa for E, mm for R, kN for F
           -> radius/depth in mm, stiffness in kN/mm.
    """
    Estar = 1.0 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    delta = (9 * F**2 / (16 * R * Estar**2)) ** (1 / 3)
    a_c = np.sqrt(R * delta)
    k_contact = 2 * Estar * np.sqrt(R * delta)
    return a_c, delta, k_contact


# =====================================================================
# 4. Combined model
# =====================================================================

def predict_panel_stiffness(D, layup_h, a, b, F,
                             G13, G23,
                             R_indenter, E_indenter, nu_indenter,
                             E_surface, nu_surface,
                             k_shear=5 / 6, M=241, N=241):
    """
    Full pipeline: shear-deformable plate (loaded over the Hertz contact
    patch at force F) in series with Hertz contact compliance.

    Returns dict with a_c, k_plate, k_contact, k_total, w_total (at load F).
    """
    A44, A55 = transverse_shear_stiffness(G13, G23, layup_h, k_shear)
    a_c, delta, k_contact = hertz_contact(F, R_indenter, E_indenter, nu_indenter,
                                           E_surface, nu_surface)
    k_plate = plate_stiffness(D, A44, A55, a, b, a_c, P_ref=1.0, M=M, N=N)
    k_total = 1.0 / (1.0 / k_plate + 1.0 / k_contact)
    return {
        "a_c_mm": a_c,
        "indentation_mm": delta,
        "k_plate_kN_mm": k_plate,
        "k_contact_kN_mm": k_contact,
        "k_total_kN_mm": k_total,
        "deflection_at_F_mm": F / k_total,
    }


# =====================================================================
# Demo: QIT / DD30 / DD45 panels from the original problem
# =====================================================================

if __name__ == "__main__":
    # --- Ply properties (UD CFRP, from Table 1) ---
    E11, E22, nu12, G12 = 115.0, 8.2, 0.34, 3.6   # GPa
    G13, G23 = 3.6, 3.6                            # GPa (transversely isotropic in shear)
    Q = ply_Q_matrix(E11, E22, nu12, G12)

    t_ply = 0.2875  # mm
    h_laminate = 16 * t_ply  # 4.6 mm

    layups = {
        "QIT":  [45, 45, -45, -45, 0, 0, 90, 90, 90, 90, 0, 0, -45, -45, 45, 45],
        "DD30": [30, -60, -30, 60] * 4,
        "DD45": [45, -45, -45, 45] * 4,
    }

    # --- Test geometry ---
    a_window, b_window = 125.0, 75.0  # mm, clamped window (simply-supported span)

    # --- Indenter ---
    R_ball = 8.0                # mm (16 mm diameter hemisphere)
    E_steel, nu_steel = 200.0, 0.3   # GPa

    # --- Experimental reference point ---
    F_test = 4.0     # kN
    w_test = 1.25    # mm
    k_test = F_test / w_test

    print(f"Experimental: {F_test} kN at {w_test} mm  ->  secant stiffness = {k_test:.3f} kN/mm\n")
    print(f"{'Laminate':8s} {'D11':>7s} {'D22':>7s} {'D12':>7s} {'D66':>7s} "
          f"{'a_c(mm)':>8s} {'k_plate':>8s} {'k_contact':>10s} {'k_total':>8s} {'w@'+str(F_test)+'kN':>10s}")

    for name, layup in layups.items():
        A, B, D = ABD_matrices(Q, layup, t_ply)
        result = predict_panel_stiffness(
            D, h_laminate, a_window, b_window, F_test,
            G13, G23,
            R_ball, E_steel, nu_steel,
            E22, nu12,   # composite surface modulus/Poisson approx (E33=E22, nu13=nu12)
        )
        print(f"{name:8s} {D[0,0]:7.1f} {D[1,1]:7.1f} {D[0,1]:7.1f} {D[2,2]:7.1f} "
              f"{result['a_c_mm']:8.3f} {result['k_plate_kN_mm']:8.3f} "
              f"{result['k_contact_kN_mm']:10.2f} {result['k_total_kN_mm']:8.3f} "
              f"{result['deflection_at_F_mm']:10.3f}")

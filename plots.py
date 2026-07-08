"""
plots.py

Generates three sets of plots from the panel_stiffness model:

1. Load-deflection curves  (model vs experimental data point)
2. Deflection contour/shape of the plate under load
3. Sensitivity plots (effective span, contact radius/load, laminate comparison)

Run with: uv run plots.py
Output: PNG files written to ./figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panel_stiffness import (
    ply_Q_matrix, ABD_matrices, transverse_shear_stiffness,
    hertz_contact, plate_stiffness, predict_panel_stiffness,
    fsdt_deflection_field,
)

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------
# Shared setup (same as panel_stiffness.py __main__ block)
# ---------------------------------------------------------------
E11, E22, nu12, G12 = 115.0, 8.2, 0.34, 3.6
G13, G23 = 3.6, 3.6
Q = ply_Q_matrix(E11, E22, nu12, G12)
t_ply = 0.2875
h_laminate = 16 * t_ply

layups = {
    "QIT":  [45, 45, -45, -45, 0, 0, 90, 90, 90, 90, 0, 0, -45, -45, 45, 45],
    "DD30": [30, -60, -30, 60] * 4,
    "DD45": [45, -45, -45, 45] * 4,
}
colors = {"QIT": "tab:blue", "DD30": "tab:orange", "DD45": "tab:green"}

a_window, b_window = 125.0, 75.0
R_ball, E_steel, nu_steel = 8.0, 200.0, 0.3

D_of = {}
for name, lay in layups.items():
    _, _, D = ABD_matrices(Q, lay, t_ply)
    D_of[name] = D

F_test, w_test = 4.0, 1.25
k_test = F_test / w_test

# =================================================================
# 1. Load-deflection curves
# =================================================================
F_range = np.linspace(0.05, 4.7, 25)

plt.figure(figsize=(7, 5.5))
for name, D in D_of.items():
    w_vals = []
    for F in F_range:
        r = predict_panel_stiffness(D, h_laminate, a_window, b_window, F,
                                     G13, G23, R_ball, E_steel, nu_steel,
                                     E22, nu12, M=161, N=161)
        w_vals.append(r["deflection_at_F_mm"])
    plt.plot(w_vals, F_range, "-o", ms=3, color=colors[name], label=f"{name} (model)")

# experimental: straight line through origin, slope k_test, up to 4.7 kN
w_exp = np.array([0, F_test / k_test * (4.7 / F_test)])
F_exp = np.array([0, 4.7])
plt.plot(F_exp / k_test, F_exp, "k--", lw=2, label="Experimental (3.20 kN/mm, linear)")
plt.plot([w_test], [F_test], "kx", ms=10, mew=2, label="Measured point (4 kN, 1.25 mm)")

plt.xlabel("Central deflection (mm)")
plt.ylabel("Load (kN)")
plt.title("Load-deflection: model vs experiment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/load_deflection.png", dpi=150)
plt.close()

# =================================================================
# 2. Deflection contour / shape at F_test, for each laminate
# =================================================================
x = np.linspace(0, a_window, 90)
y = np.linspace(0, b_window, 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
vmax = 0
fields = {}
for name, D in D_of.items():
    A44, A55 = transverse_shear_stiffness(G13, G23, h_laminate)
    a_c, _, _ = hertz_contact(F_test, R_ball, E_steel, nu_steel, E22, nu12)
    field = fsdt_deflection_field(D, A44, A55, a_window, b_window, F_test, a_c, x, y, M=81, N=61)
    fields[name] = field
    vmax = max(vmax, field.max())

for ax, (name, field) in zip(axes, fields.items()):
    im = ax.contourf(x, y, field.T, levels=20, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(f"{name}: w at {F_test} kN (max {field.max():.3f} mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, label="deflection (mm)", shrink=0.8)
plt.savefig(f"{OUTDIR}/deflection_contours.png", dpi=150)
plt.close()

# =================================================================
# 3a. Sensitivity: stiffness vs assumed effective span
# =================================================================
span_fracs = np.linspace(0.0, 1.0, 11)  # 0 = window, 1 = full panel
a_full, b_full = 150.0, 100.0

plt.figure(figsize=(7, 5.5))
for name, D in D_of.items():
    ks = []
    for f in span_fracs:
        a = a_window + f * (a_full - a_window)
        b = b_window + f * (b_full - b_window)
        r = predict_panel_stiffness(D, h_laminate, a, b, F_test,
                                     G13, G23, R_ball, E_steel, nu_steel,
                                     E22, nu12, M=161, N=161)
        ks.append(r["k_total_kN_mm"])
    plt.plot(span_fracs, ks, "-o", ms=4, color=colors[name], label=name)
plt.axhline(k_test, color="k", ls="--", label="Experimental (3.20 kN/mm)")
plt.xlabel("Fraction of the way from clamped window (0) to full panel (1)")
plt.ylabel("Predicted k_total (kN/mm)")
plt.title("Sensitivity to effective simply-supported span")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/sensitivity_span.png", dpi=150)
plt.close()

# =================================================================
# 3b. Sensitivity: k_plate, k_contact, k_total vs load
# =================================================================
F_range2 = np.linspace(0.1, 4.7, 20)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
for name, D in D_of.items():
    kps, kcs, kts = [], [], []
    for F in F_range2:
        r = predict_panel_stiffness(D, h_laminate, a_window, b_window, F,
                                     G13, G23, R_ball, E_steel, nu_steel,
                                     E22, nu12, M=161, N=161)
        kps.append(r["k_plate_kN_mm"])
        kcs.append(r["k_contact_kN_mm"])
        kts.append(r["k_total_kN_mm"])
    axes[0].plot(F_range2, kps, "-o", ms=3, color=colors[name], label=name)
    axes[1].plot(F_range2, kcs, "-o", ms=3, color=colors[name], label=name)
    axes[2].plot(F_range2, kts, "-o", ms=3, color=colors[name], label=name)

for ax, title in zip(axes, ["Plate-only k_plate", "Contact-only k_contact", "Combined k_total"]):
    ax.axhline(k_test, color="k", ls="--", lw=1, label="Experimental" if title == "Combined k_total" else None)
    ax.set_xlabel("Load (kN)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Stiffness (kN/mm)")
axes[0].legend()
axes[2].legend()
plt.savefig(f"{OUTDIR}/sensitivity_load_breakdown.png", dpi=150)
plt.close()

# =================================================================
# 3c. Laminate comparison bar chart at the test load
# =================================================================
plt.figure(figsize=(6, 5))
names = list(D_of.keys())
k_plates, k_contacts, k_totals = [], [], []
for name in names:
    r = predict_panel_stiffness(D_of[name], h_laminate, a_window, b_window, F_test,
                                 G13, G23, R_ball, E_steel, nu_steel,
                                 E22, nu12, M=241, N=241)
    k_plates.append(r["k_plate_kN_mm"])
    k_contacts.append(r["k_contact_kN_mm"])
    k_totals.append(r["k_total_kN_mm"])

xpos = np.arange(len(names))
width = 0.25
plt.bar(xpos - width, k_plates, width, label="k_plate")
plt.bar(xpos, k_contacts, width, label="k_contact")
plt.bar(xpos + width, k_totals, width, label="k_total")
plt.axhline(k_test, color="k", ls="--", label="Experimental (3.20 kN/mm)")
plt.xticks(xpos, names)
plt.ylabel("Stiffness (kN/mm)")
plt.title(f"Stiffness contributions at {F_test} kN")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/laminate_comparison_bar.png", dpi=150)
plt.close()

print(f"Wrote plots to ./{OUTDIR}/:")
for f in sorted(os.listdir(OUTDIR)):
    print(" -", f)

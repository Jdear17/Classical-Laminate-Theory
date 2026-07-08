# Panel stiffness: D-matrix / indentation study

Predicts the out-of-plane (central load / central deflection) stiffness of a
composite laminate panel under a hemispherical indenter, and compares it
against experimental data for three layups: **QIT**, **DD30**, **DD45**.

## Background

`QIT and DD.xlsx` computes the CLT D-matrix (D11 in particular) for the three
laminates. `Information for Claude AI.docx` poses the underlying question:
D11 varies by ~19% across the three laminates (QIT 360, DD30 345, DD45 292
kN·mm), but all three measure almost the same experimental panel stiffness
(~3.2 kN/mm, from a 4 kN load at 1.25 mm deflection, linear to 4.7 kN). Why?

Panel test setup: a 150×100 mm panel, clamped over a 125×75 mm window, loaded
centrally with a 16 mm diameter hemispherical stainless-steel impactor.

## Status

Complete. `panel_stiffness.py` and `plots.py` both run cleanly and reproduce
the validated numbers below. The model has been checked against standard CLT
and Hertz-contact theory (formulas re-derived independently, not just
run), and the one code issue found — a stale docstring reference to a
`demo_span_sensitivity()` function that was never written — has been fixed.

## Findings

1. **The Excel D11 values check out.** Independently rebuilt in Python
   (ply `Q` → rotation → through-thickness integration) and match the
   spreadsheet to 3 significant figures.
2. **D11 alone doesn't predict panel stiffness.** Central deflection under a
   patch/point load depends on the full D matrix (D11, D22, D12, D66), not
   D11 in isolation. D22 and D66 largely offset the D11 spread across the
   three laminates.
3. **Classical (Kirchhoff) thin-plate theory overpredicts stiffness** by
   55–65% versus the measured ~3.2 kN/mm. Two corrections close the gap:
   - **Transverse shear deformation** (Reissner-Mindlin/FSDT) — this ply's
     transverse shear modulus is low relative to E11, so the laminate is
     shear-soft through its thickness.
   - **Hertzian contact compliance** of the 16 mm steel hemisphere against
     the composite surface, in series with the plate's own bending/shear
     compliance (`1/k_total = 1/k_plate + 1/k_contact`). At the 4 kN test
     point this contact spring is the same order of magnitude as the plate
     stiffness itself, and it's the main reason the three laminates end up
     reading so close together experimentally.
4. **Effective span matters more than expected.** Modelling the simply
   supported span as the clamped **window** (125×75 mm) — not the full
   150×100 mm panel — is what makes the model match the data. This implies
   the toggle clamps are acting as a true simple support at the window edge,
   with negligible effective stiffness contribution from the overhang.

With window span + patch-loaded FSDT + Hertz contact in series, predicted
stiffness at 4 kN lands at 3.36–3.51 kN/mm across the three laminates,
against a measured 3.20 kN/mm (within ~5–10%).

## How it works

```
panel_stiffness.py
├── 1. Classical Lamination Theory (CLT)
│      ply_Q_matrix / rotate_Q / ABD_matrices  -> full A, B, D per laminate
├── 2. First-order Shear Deformation Theory (Reissner-Mindlin)
│      transverse_shear_stiffness, patch_load_fourier_coeff,
│      fsdt_center_deflection / fsdt_deflection_field, plate_stiffness
│        -> Navier series solution for a simply supported rectangular
│           plate, loaded over a small patch (not a mathematical point,
│           which is singular in FSDT)
├── 3. Hertz contact theory
│      hertz_contact  -> local indentation stiffness of the steel
│                         hemisphere against the composite surface
└── 4. Series combination
       predict_panel_stiffness  -> 1/k_total = 1/k_plate + 1/k_contact
```

`plots.py` runs the same pipeline and writes five figures to `figures/`:
load-deflection curves, deflection contour shapes, span sensitivity, a
load-vs-stiffness-contribution breakdown, and a laminate comparison bar
chart.

## Assumptions

- Ply properties: E11=115 GPa, E22=8.2 GPa, ν12=0.34, G12=3.6 GPa,
  t_ply=0.2875 mm (16-ply laminates, h=4.6 mm), per Table 1 of the source doc.
- Transverse shear moduli G13=G23=3.6 GPa (transversely isotropic ply).
- Simply-supported span = the 125×75 mm clamped window, not the 150×100 mm
  full panel (justified by the span-sensitivity fit above).
- Contact patch radius for the plate load is the Hertz contact radius at the
  applied force (not the 8 mm ball radius itself — the actual elastic
  contact patch is sub-mm).
- Hertz contact treats the composite surface as an isotropic half-space
  using E22 (=E33) and ν12 (=ν13) as a stand-in for the true anisotropic
  surface stiffness.
- Shear correction factor k=5/6 (generic Reissner-Mindlin value, not
  measured for this layup).

## Limitations

- **D16/D26 (bending-twist coupling) and any B matrix (bending-stretching
  coupling) are dropped** from the FSDT solve. The Navier solution used here
  strictly requires a specially-orthotropic plate (D16=D26=0, B=0). QIT and
  DD45 as tested are exact stacking palindromes (B=0 in reality), but DD30's
  real stacking (30/-60/-30/60 repeated, not mirrored) is **not** symmetric
  and has a non-zero B11 (~18 kN) — a second-order effect on D11 but ignored
  here. D16/D26 are 3–20% of D11 for these laminates, a reasonable but not
  exact approximation. `ABD_matrices()` already returns the full B and D16/D26
  terms if this needs generalising (would require more Navier series unknowns
  per mode).
- **Hertz contact assumes isotropic bodies**; the composite's true
  anisotropic transverse response isn't captured, only approximated via E22.
- **No fixture/frame compliance term.** The model assumes the clamped window
  is a rigid, ideal simple support. Remaining ~5–10% gap between predicted
  and measured deflection is most plausibly a mix of this, the isotropic
  Hertz approximation, and the generic shear correction factor — roughly in
  that order of likely contribution.
- A true point load does not converge in FSDT (mathematical singularity);
  the patch-load formulation avoids this but still assumes a *square* patch
  of the Hertz contact radius, not the true circular Hertzian pressure
  distribution.

## Running

```
uv sync
uv run panel_stiffness.py   # prints the QIT/DD30/DD45 comparison table
uv run plots.py              # writes figures/*.png
```

## Files

- `panel_stiffness.py` — model (CLT + FSDT Navier solve + Hertz contact)
- `plots.py` — figure generation
- `figures/` — rendered PNGs (load-deflection, deflection contours, span
  sensitivity, load breakdown, laminate comparison)
- `QIT and DD.xlsx` — original D-matrix calculation
- `Information for Claude AI.docx` — original problem statement

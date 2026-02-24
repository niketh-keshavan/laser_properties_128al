"""
Beam Profile Analysis — Exercises IV.8 through IV.12
Knife-edge measurement of a Gaussian laser beam.

Micrometer calibration: 1 division = 10 µm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import interp1d

# ── Micrometer calibration ─────────────────────────────────────────────────
UM_PER_DIV = 10.0  # 1 micrometer division = 10 µm

# ═══════════════════════════════════════════════════════════════════════════
# Exercise IV.8  —  Gaussian beam intensity & total power
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Exercise IV.8 — Gaussian Beam Intensity")
print("=" * 65)
print("""
For a TEM₀₀ Gaussian beam propagating along z, the intensity is:

    I(x, y) = I₀ · exp( −2(x² + y²) / w² )

where
    I₀  = peak (on-axis) intensity,
    w   = beam radius (1/e² intensity radius),
    r   = √(x² + y²) is the perpendicular distance from the axis.

Total power (integrate over the full transverse plane):

    P_total = ∫∫ I₀ exp(−2r²/w²) dx dy
            = I₀ · (π w² / 2)

Solving for I₀:  I₀ = 2 P_total / (π w²)
""")

# ═══════════════════════════════════════════════════════════════════════════
# Exercise IV.9  —  Load knife-edge data & plot P(x)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Exercise IV.9 — Knife-Edge Beam Profile Measurement")
print("=" * 65)

df = pd.read_csv("data/experiment_B.csv")

# Convert micrometer dial divisions → µm
x_pos_div = df["Position (uM)"].values.astype(float)
x_pos = x_pos_div * UM_PER_DIV  # now in µm

power_min = df["Power Min (uW)"].values.astype(float)
power_max = df["Power Max (uW)"].values.astype(float)
power_adj = df["Power Adjusted (uW)"].values.astype(float)  # background already subtracted

# Uncertainty: half-range of min/max; floor at 0.3 µW
power_unc = (power_max - power_min) / 2.0
power_unc = np.maximum(power_unc, 0.3)

# Sort by increasing position
idx = np.argsort(x_pos)
x_pos = x_pos[idx]
power_adj = power_adj[idx]
power_unc = power_unc[idx]

# Also store in mm for convenient display
x_pos_mm = x_pos / 1000.0

print(f"\n  Micrometer calibration: 1 div = {UM_PER_DIV} µm")
print(f"  Positions range : {x_pos[0]:.0f}  to  {x_pos[-1]:.0f} µm  "
      f"({x_pos_mm[0]:.1f} to {x_pos_mm[-1]:.1f} mm)")
print(f"  Power range     : {power_adj[0]:.2f}  to  {power_adj[-1]:.2f} µW")

# ── Plot raw P(x) ──────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(x_pos_mm, power_adj, yerr=power_unc,
             fmt="o", capsize=3, color="royalblue", label="Measured P(x)")
ax1.set_xlabel("Knife-Edge Position (mm)", fontsize=13)
ax1.set_ylabel("Power (µW)", fontsize=13)
ax1.set_title("Exercise IV.9 — Knife-Edge Beam Profile P(x)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("ex_IV9_beam_profile.png", dpi=200)
print("\n  Saved  ex_IV9_beam_profile.png")

# ═══════════════════════════════════════════════════════════════════════════
# Exercise IV.10  —  Theoretical knife-edge expression & fit
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Exercise IV.10 — Theoretical Knife-Edge Expression")
print("=" * 65)
print("""
When a knife edge at position x blocks the beam for x' < x,
the transmitted power is:

    P(x) = ∫_x^∞ ∫_{-∞}^{∞} I₀ exp(−2(x'² + y'²)/w²) dy' dx'

The y-integral gives √(π/2) · w.  The x-integral gives:

    P(x) = (P_total / 2) · erfc( √2 · (x − x₀) / w )
         = (P_total / 2) · [ 1 + erf( √2 · (x₀ − x) / w ) ]

Equivalently (knife edge moving in from the +x side):

    P(x) = (A / 2) · [ 1 + erf( √2 (x − x₀) / w ) ]

where A = total unblocked power, x₀ = beam center, w = 1/e² radius.
""")


# ── Fit model (works in µm) ───────────────────────────────────────────────
def knife_edge_model(x, A, x0, w):
    """Transmitted power past a knife edge for a Gaussian beam."""
    return (A / 2.0) * (1.0 + erf(np.sqrt(2) * (x - x0) / w))


# Initial guesses (in µm)
A0 = np.max(power_adj)
x0_guess = x_pos[np.argmin(np.abs(power_adj - A0 / 2))]
w0_guess = 200.0  # µm

popt, pcov = curve_fit(
    knife_edge_model, x_pos, power_adj,
    p0=[A0, x0_guess, w0_guess],
    sigma=power_unc, absolute_sigma=True,
)
perr = np.sqrt(np.diag(pcov))

A_fit, x0_fit, w_fit = popt
A_err, x0_err, w_err = perr

# Convert to mm for display
w_mm = w_fit / 1000.0
w_mm_err = w_err / 1000.0
x0_mm = x0_fit / 1000.0
x0_mm_err = x0_err / 1000.0

print("  Fit results:")
print(f"    A  (total power)  = {A_fit:8.2f} ± {A_err:.2f} µW")
print(f"    x₀ (beam centre)  = {x0_fit:8.1f} ± {x0_err:.1f} µm  ({x0_mm:.4f} ± {x0_mm_err:.4f} mm)")
print(f"    w  (1/e² radius)  = {w_fit:8.1f} ± {w_err:.1f} µm  ({w_mm:.4f} ± {w_mm_err:.4f} mm)")

# Reduced chi-squared
residuals = power_adj - knife_edge_model(x_pos, *popt)
chi2 = np.sum((residuals / power_unc) ** 2)
ndof = len(x_pos) - len(popt)
print(f"    χ²/ndf            = {chi2:.2f} / {ndof}  =  {chi2 / ndof:.2f}")

# ── Plot data + fit (x-axis in mm) ────────────────────────────────────────
x_smooth = np.linspace(x_pos[0] - 200, x_pos[-1] + 200, 500)
x_smooth_mm = x_smooth / 1000.0
P_smooth = knife_edge_model(x_smooth, *popt)

# ── Propagate fit-parameter uncertainty → confidence band ─────────────────
# Compute Jacobian of the model w.r.t. (A, x0, w) at each smooth point
def _model_jacobian(x, A, x0, w):
    """Partial derivatives of knife_edge_model w.r.t. A, x0, w."""
    arg = np.sqrt(2) * (x - x0) / w
    erf_val = erf(arg)
    gauss = np.exp(-arg**2)           # = exp(-2(x-x0)^2/w^2)
    dP_dA  = 0.5 * (1.0 + erf_val)
    dP_dx0 = -A / (np.sqrt(np.pi) * w) * gauss * np.sqrt(2)
    dP_dw  = -A * np.sqrt(2) * (x - x0) / (np.sqrt(np.pi) * w**2) * gauss
    return np.column_stack([dP_dA, dP_dx0, dP_dw])

J = _model_jacobian(x_smooth, *popt)               # (N, 3)
P_var = np.einsum("ij,jk,ik->i", J, pcov, J)       # σ² at each point
P_sigma = np.sqrt(np.maximum(P_var, 0.0))           # 1-σ band

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.fill_between(x_smooth_mm, P_smooth - P_sigma, P_smooth + P_sigma,
                 color="crimson", alpha=0.18, label="1σ confidence band")
ax2.errorbar(x_pos_mm, power_adj, yerr=power_unc,
             fmt="o", capsize=4, markersize=6, color="royalblue",
             elinewidth=1.2, label="Data")
ax2.plot(x_smooth_mm, P_smooth, "-", color="crimson", linewidth=2,
         label=(f"Fit: $x_0$={x0_mm:.3f}±{x0_mm_err:.3f} mm, "
                f"$w$={w_mm:.3f}±{w_mm_err:.3f} mm"))
ax2.axvline(x0_mm, ls="--", color="gray", alpha=0.6,
            label=f"Beam centre $x_0$={x0_mm:.3f}±{x0_mm_err:.3f} mm")
ax2.set_xlabel("Knife-Edge Position (mm)", fontsize=13)
ax2.set_ylabel("Power (µW)", fontsize=13)
ax2.set_title("Exercise IV.10 — Error-Function Fit to Knife-Edge Data", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("ex_IV10_erf_fit.png", dpi=200)
print("\n  Saved  ex_IV10_erf_fit.png")

# ── Residuals plot ─────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 2.5))
ax3.errorbar(x_pos_mm, residuals, yerr=power_unc,
             fmt="o", capsize=3, color="royalblue")
ax3.axhline(0, color="crimson", linewidth=1.5)
ax3.set_xlabel("Knife-Edge Position (mm)", fontsize=13)
ax3.set_ylabel("Residual (µW)", fontsize=13)
ax3.set_title("Residuals (Data − Fit)", fontsize=14)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("ex_IV10_residuals.png", dpi=200)
print("  Saved  ex_IV10_residuals.png")

# ═══════════════════════════════════════════════════════════════════════════
# Exercise IV.11  —  Beam diameter & power fraction
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Exercise IV.11 — Beam Diameter & Power Fraction")
print("=" * 65)
print("""
The beam diameter is defined where the ELECTRIC FIELD drops to 1/e
of its maximum:
    E(r) = E₀ exp(−r²/w²)   →   E = E₀/e  at  r = w

So the beam diameter  d = 2w.

The fraction of total power inside r = w:
    F(w) = 1 − exp(−2w²/w²) = 1 − e⁻² ≈ 86.47 %
""")

beam_diameter = 2.0 * w_fit          # µm
beam_diameter_err = 2.0 * w_err      # µm
beam_diameter_mm = beam_diameter / 1000.0
beam_diameter_mm_err = beam_diameter_err / 1000.0

print(f"  Beam radius  w       = {w_fit:.1f} ± {w_err:.1f} µm  ({w_mm:.4f} ± {w_mm_err:.4f} mm)")
print(f"  Beam diameter 2w     = {beam_diameter:.1f} ± {beam_diameter_err:.1f} µm  "
      f"({beam_diameter_mm:.3f} ± {beam_diameter_mm_err:.3f} mm)")
print()

frac_inside = 1.0 - np.exp(-2)
print(f"  Fraction of power inside beam diameter (r ≤ w):")
print(f"    F = 1 − e⁻² = {frac_inside:.4f}  ({frac_inside * 100:.2f} %)")
print(f"    Power inside = {frac_inside * A_fit:.2f} µW  (out of {A_fit:.2f} µW total)")

# ── Reconstructed Gaussian intensity profile (derivative of P(x)) ─────────
Ix_smooth = (A_fit * np.sqrt(2) / (np.sqrt(np.pi) * w_fit)) * \
    np.exp(-2.0 * (x_smooth - x0_fit) ** 2 / w_fit ** 2)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(x_smooth_mm, Ix_smooth, "-", color="darkorange", linewidth=2,
         label="Reconstructed beam profile  dP/dx")
ax4.axvline((x0_fit - w_fit) / 1000, ls="--", color="green", alpha=0.7,
            label=f"$x_0 - w$ = {(x0_fit - w_fit)/1000:.3f} mm")
ax4.axvline((x0_fit + w_fit) / 1000, ls="--", color="green", alpha=0.7,
            label=f"$x_0 + w$ = {(x0_fit + w_fit)/1000:.3f} mm")
ax4.axvline(x0_mm, ls=":", color="gray", alpha=0.5,
            label=f"Centre $x_0$ = {x0_mm:.3f} mm")

# Shade region within beam diameter
mask_inside = (x_smooth >= x0_fit - w_fit) & (x_smooth <= x0_fit + w_fit)
ax4.fill_between(x_smooth_mm, Ix_smooth, where=mask_inside,
                 alpha=0.25, color="green",
                 label=f"Inside 2w ({frac_inside*100:.1f}% power)")

ax4.set_xlabel("Position (mm)", fontsize=13)
ax4.set_ylabel("Intensity (arb. units)", fontsize=13)
ax4.set_title("Exercise IV.11 — Reconstructed Gaussian Profile & Beam Diameter", fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig("ex_IV11_beam_diameter.png", dpi=200)
print("\n  Saved  ex_IV11_beam_diameter.png")

# ═══════════════════════════════════════════════════════════════════════════
# Exercise IV.12  —  Quick beam-diameter method using two power readings
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Exercise IV.12 — Quick Beam Diameter from Two Measurements")
print("=" * 65)
print("""
Theory:  P(x) = (A/2) [1 + erf(√2 (x − x₀)/w)]

    • Knife edge at x = x₀ + w  (one beam radius *beyond* centre):
        P/A = ½ [1 + erf(√2)] = ½ (1 + 0.9545) = 0.9772   → 97.72 %

    • Knife edge at x = x₀ − w  (one beam radius *inside* centre):
        P/A = ½ [1 − erf(√2)] = ½ (1 − 0.9545) = 0.0228   → 2.28 %

Method: Move knife edge to the position where P = 0.9772 A,
        then to P = 0.0228 A.  The distance between these two
        positions is 2w = beam diameter.
""")

# Theoretical fractions
frac_plus  = 0.5 * (1 + erf(np.sqrt(2)))   # x = x0 + w
frac_minus = 0.5 * (1 - erf(np.sqrt(2)))   # x = x0 - w
print(f"  Theoretical power fractions:")
print(f"    At x₀ + w :  P/A = {frac_plus:.6f}  ({frac_plus*100:.2f} %)")
print(f"    At x₀ − w :  P/A = {frac_minus:.6f}  ({frac_minus*100:.2f} %)")
print(f"    At x₀     :  P/A = 0.5000    (50.00 %)")
print()

# Corresponding power values from our fit
P_at_plus  = frac_plus  * A_fit
P_at_minus = frac_minus * A_fit
P_at_center = 0.5 * A_fit

x_plus_mm  = (x0_fit + w_fit) / 1000
x_minus_mm = (x0_fit - w_fit) / 1000

print(f"  For our beam (A = {A_fit:.2f} µW):")
print(f"    Move knife edge to P = {P_at_plus:.2f} µW  → x₀ + w = {x_plus_mm:.3f} mm")
print(f"    Move knife edge to P = {P_at_minus:.2f} µW  → x₀ − w = {x_minus_mm:.3f} mm")
print(f"    Distance = 2w = {beam_diameter:.1f} µm = {beam_diameter_mm:.3f} mm  "
      f"(± {beam_diameter_mm_err:.3f} mm)")
print()

# Also compute from data by interpolation
interp_func = interp1d(power_adj, x_pos, kind="linear", fill_value="extrapolate")
x_at_9772 = float(interp_func(frac_plus * A_fit))
x_at_0228 = float(interp_func(frac_minus * A_fit))
beam_diam_quick = x_at_9772 - x_at_0228

print(f"  From data interpolation:")
print(f"    Position at {frac_plus*100:.2f}% power  ≈ {x_at_9772:.0f} µm ({x_at_9772/1000:.3f} mm)")
print(f"    Position at {frac_minus*100:.2f}% power  ≈ {x_at_0228:.0f} µm ({x_at_0228/1000:.3f} mm)")
print(f"    Beam diameter (2w) quick method = {beam_diam_quick:.0f} µm = {beam_diam_quick/1000:.3f} mm")
print(f"    Beam diameter (2w) full fit     = {beam_diameter:.1f} µm = {beam_diameter_mm:.3f} mm "
      f"± {beam_diameter_mm_err:.3f} mm")

# ── Summary plot with annotated positions (mm) ────────────────────────────
fig5, ax5 = plt.subplots(figsize=(9, 5.5))
ax5.errorbar(x_pos_mm, power_adj, yerr=power_unc,
             fmt="o", capsize=3, color="royalblue", label="Data", zorder=3)
ax5.plot(x_smooth_mm, P_smooth, "-", color="crimson", linewidth=2, label="Erf fit", zorder=2)

# Mark key positions
ax5.axhline(A_fit, ls=":", color="gray", alpha=0.5)
ax5.axhline(A_fit / 2, ls=":", color="gray", alpha=0.5)

ax5.axvline(x0_mm, ls="--", color="gray", alpha=0.6)
ax5.axvline(x_minus_mm, ls="--", color="green", alpha=0.7)
ax5.axvline(x_plus_mm, ls="--", color="green", alpha=0.7)

# Annotate
ax5.annotate(f"$x_0 - w$ = {x_minus_mm:.3f} mm\n({frac_minus*100:.1f}% power)",
             xy=(x_minus_mm, P_at_minus), fontsize=9,
             xytext=(-60, 40), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="green"),
             color="green", fontweight="bold")
ax5.annotate(f"$x_0$ = {x0_mm:.3f} mm\n(50% power)",
             xy=(x0_mm, A_fit / 2), fontsize=9,
             xytext=(15, -30), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="gray"),
             color="gray", fontweight="bold")
ax5.annotate(f"$x_0 + w$ = {x_plus_mm:.3f} mm\n({frac_plus*100:.1f}% power)",
             xy=(x_plus_mm, P_at_plus), fontsize=9,
             xytext=(15, -40), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="green"),
             color="green", fontweight="bold")

# Double arrow for beam diameter
y_arrow = A_fit * 0.15
ax5.annotate("", xy=(x_plus_mm, y_arrow), xytext=(x_minus_mm, y_arrow),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
ax5.text(x0_mm, y_arrow + A_fit * 0.04,
         f"2w = {beam_diameter_mm:.3f} mm", ha="center", fontsize=11,
         color="purple", fontweight="bold")

ax5.set_xlabel("Knife-Edge Position (mm)", fontsize=13)
ax5.set_ylabel("Power (µW)", fontsize=13)
ax5.set_title("Exercise IV.12 — Quick Beam Diameter Method", fontsize=14)
ax5.legend(fontsize=10, loc="center left")
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig("ex_IV12_quick_diameter.png", dpi=200)
print("\n  Saved  ex_IV12_quick_diameter.png")

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SUMMARY  (1 micrometer division = 10 µm)")
print("=" * 65)
print(f"  Total beam power  A    = {A_fit:.2f} ± {A_err:.2f} µW")
print(f"  Beam centre       x₀   = {x0_fit:.0f} ± {x0_err:.0f} µm  ({x0_mm:.3f} mm)")
print(f"  1/e² beam radius  w    = {w_fit:.1f} ± {w_err:.1f} µm  ({w_mm:.4f} mm)")
print(f"  Beam diameter     2w   = {beam_diameter:.1f} ± {beam_diameter_err:.1f} µm  "
      f"({beam_diameter_mm:.3f} ± {beam_diameter_mm_err:.3f} mm)")
print(f"  Power inside 2w        = {frac_inside*100:.2f} %")
print(f"  χ²/ndf                 = {chi2/ndof:.2f}")
print("=" * 65)

plt.show()

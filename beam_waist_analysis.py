"""
Beam Waist & Depth of Field Analysis
Fits the Gaussian beam propagation model to experiment_D.csv data
to extract D (beam waist) and L (depth of field).

Model:  D(z) = D0 * sqrt(1 + ((z - z0) / zR)^2)

where   D0 = beam waist diameter (minimum),
        z0 = waist position,
        zR = Rayleigh range,
        L  = 2 * zR  (depth of field).
"""

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no Tk needed)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/experiment_D.csv")

z = df["Distance (m)"].values          # metres
D = df["Raw Delta x (mm)"].values      # mm  (raw knife-edge delta)

print("=" * 60)
print("  Beam Waist & Depth of Field Analysis")
print("=" * 60)
print(f"\n  Data points: {len(z)}")
print(f"  z range:     {z.min():.2f} – {z.max():.2f} m")
print(f"  D range:     {D.min():.2f} – {D.max():.2f} mm")


# ── Gaussian beam propagation model ───────────────────────────────────────
def beam_diameter(z, D0, z0, zR):
    """Beam diameter vs propagation distance for a Gaussian beam."""
    return D0 * np.sqrt(1.0 + ((z - z0) / zR) ** 2)


# ── Initial guesses ───────────────────────────────────────────────────────
# Minimum diameter is near z ≈ 0.14 m from the data
D0_guess = D.min()
z0_guess = z[np.argmin(D)]
zR_guess = 0.05  # metres

# ── Fit ────────────────────────────────────────────────────────────────────
popt, pcov = curve_fit(
    beam_diameter, z, D,
    p0=[D0_guess, z0_guess, zR_guess],
)
perr = np.sqrt(np.diag(pcov))

D0_fit, z0_fit, zR_fit = popt
D0_err, z0_err, zR_err = perr

# Depth of field
L_fit = 2.0 * abs(zR_fit)
L_err = 2.0 * zR_err

# ── Results ────────────────────────────────────────────────────────────────
print("\n  Fit results (Gaussian beam propagation):")
print(f"    D  (beam waist)      = {D0_fit:.3f} ± {D0_err:.3f} mm")
print(f"    z₀ (waist position)  = {z0_fit:.4f} ± {z0_err:.4f} m")
print(f"    zR (Rayleigh range)  = {abs(zR_fit):.4f} ± {zR_err:.4f} m")
print(f"    L  (depth of field)  = {L_fit:.4f} ± {L_err:.4f} m")
print(f"                         = {L_fit*100:.2f} ± {L_err*100:.2f} cm")

# Residuals
residuals = D - beam_diameter(z, *popt)
ndof = len(z) - len(popt)
rms_res = np.sqrt(np.mean(residuals ** 2))
print(f"\n    RMS residual         = {rms_res:.4f} mm")

# ── Plot 1: Beam diameter + fit ────────────────────────────────────────────
z_smooth = np.linspace(z.min() - 0.02, z.max() + 0.02, 500)
D_smooth = beam_diameter(z_smooth, *popt)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)

# ── Top panel: data + fit ──────────────────────────────────────────────────
ax1.plot(z, D, "o", color="blue", markersize=7, label="Data", zorder=3)
ax1.plot(z_smooth, D_smooth, "-", color="red", linewidth=2,
         label=(f"Fit: $D(z) = D_0\\sqrt{{1+(z-z_0)^2/z_R^2}}$\n"
                f"  $D_0$ = {D0_fit:.3f} mm,  $z_0$ = {z0_fit:.3f} m\n"
                f"  $z_R$ = {abs(zR_fit):.4f} m,  $L$ = {L_fit:.4f} m"),
         zorder=2)

# Mark beam waist
ax1.axhline(D0_fit, ls=":", color="gray", alpha=0.5)
ax1.axvline(z0_fit, ls="--", color="red", alpha=0.4,
            label=f"Waist position $z_0$ = {z0_fit:.3f} m")

# Mark depth of field (Rayleigh range boundaries)
z_left = z0_fit - abs(zR_fit)
z_right = z0_fit + abs(zR_fit)
ax1.axvspan(z_left, z_right, alpha=0.07, color="blue",
            label=f"Depth of field $L$ = {L_fit:.4f} m")

ax1.set_ylabel("Raw Beam Diameter (mm)", fontsize=13)
ax1.set_title("Beam Diameter vs Distance — Gaussian Beam Propagation", fontsize=14)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3)

# ── Bottom panel: residuals ────────────────────────────────────────────────
ax2.stem(z, residuals, linefmt="blue", markerfmt="o", basefmt="k-")
ax2.axhline(0, color="red", linewidth=1)
ax2.set_xlabel("Distance z (m)", fontsize=13)
ax2.set_ylabel("Residual (mm)", fontsize=13)
ax2.set_title("Fit Residuals", fontsize=11)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/beam_diameter_vs_z.png", dpi=200)
print(f"\n  Saved  figures/beam_diameter_vs_z.png")

# ── Plot 2: standalone residual analysis ───────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))

# Residuals vs z
ax3.plot(z, residuals, "o-", color="blue", markersize=6)
ax3.axhline(0, color="red", linewidth=1)
ax3.set_xlabel("Distance z (m)", fontsize=12)
ax3.set_ylabel("Residual (mm)", fontsize=12)
ax3.set_title("Residuals vs Distance", fontsize=12)
ax3.grid(True, alpha=0.3)

# Residual histogram
ax4.hist(residuals, bins=8, color="blue", edgecolor="red", alpha=0.7)
ax4.axvline(0, color="red", linewidth=1.5, ls="--")
ax4.set_xlabel("Residual (mm)", fontsize=12)
ax4.set_ylabel("Count", fontsize=12)
ax4.set_title(f"Residual Distribution (RMS = {rms_res:.4f} mm)", fontsize=12)
ax4.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("figures/beam_fit_residuals.png", dpi=200)
print(f"  Saved  figures/beam_fit_residuals.png")

plt.show()

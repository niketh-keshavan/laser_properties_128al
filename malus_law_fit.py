import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/experiment_A.csv")

angle_deg = df["Angle"].values.astype(float)
angle_rad = np.deg2rad(angle_deg)

power_min = df["Power_Min"].values.astype(float)
power_max = df["Power_Max"].values.astype(float)
power_avg = df["Power_Avg"].values.astype(float)

# ── Subtract background ────────────────────────────────────────────────────
background     = 0.95   # µW
background_unc = 0.05   # µW  (estimated uncertainty on background)

power_avg_corr = power_avg - background

# ── Uncertainty budget (per data point) ────────────────────────────────────
#   σ_spread   = half-range of repeated readings  (Type A)
#   σ_bg       = background subtraction uncertainty
#   σ_reading  = instrument least-count / digitisation uncertainty
#   σ_total    = quadrature sum of all three
sigma_spread  = (power_max - power_min) / 2.0
sigma_reading = 0.5   # µW – conservative estimate for meter resolution
sigma_total   = np.sqrt(sigma_spread**2 + background_unc**2 + sigma_reading**2)

print("Per-point uncertainty breakdown (µW):")
print(f"  {'Angle':>5s}  {'σ_spread':>8s}  {'σ_bg':>6s}  {'σ_read':>6s}  {'σ_total':>8s}")
for i in range(len(angle_deg)):
    print(f"  {angle_deg[i]:5.0f}  {sigma_spread[i]:8.3f}  "
          f"{background_unc:6.3f}  {sigma_reading:6.3f}  {sigma_total[i]:8.3f}")
print()

# ── Malus's Law model ──────────────────────────────────────────────────────
# P(θ) = I_max · cos²(θ − θ₀) + I_min
def malus_law(theta, I_max, I_min, theta_0):
    return I_max * np.cos(theta - theta_0) ** 2 + I_min


# ── Fit ────────────────────────────────────────────────────────────────────
p0 = [340.0, 1.0, np.deg2rad(120)]  # initial guesses
popt, pcov = curve_fit(
    malus_law, angle_rad, power_avg_corr,
    p0=p0, sigma=sigma_total, absolute_sigma=True
)
perr = np.sqrt(np.diag(pcov))  # 1-σ parameter uncertainties

I_max_fit, I_min_fit, theta0_fit = popt
I_max_err, I_min_err, theta0_err = perr

print("=" * 55)
print("         Malus's Law Fit Results")
print("=" * 55)
print(f"  I_max   = {I_max_fit:8.2f} \u00b1 {I_max_err:.2f} \u00b5W")
print(f"  I_min   = {I_min_fit:8.2f} \u00b1 {I_min_err:.2f} \u00b5W")
print(f"  \u03b8_0     = {np.rad2deg(theta0_fit):8.2f} \u00b1 {np.rad2deg(theta0_err):.2f}\u00b0")
print()

# ── Degree of polarization ─────────────────────────────────────────────────
# DOP = (I_max − I_min) / (I_max + I_min)
DOP = (I_max_fit - I_min_fit) / (I_max_fit + I_min_fit)

# Propagate uncertainty via partial derivatives
dDOP_dImax =  2 * I_min_fit / (I_max_fit + I_min_fit) ** 2
dDOP_dImin = -2 * I_max_fit / (I_max_fit + I_min_fit) ** 2
DOP_err = np.sqrt((dDOP_dImax * I_max_err) ** 2
                 + (dDOP_dImin * I_min_err) ** 2)

print(f"  Degree of polarization = {DOP:.4f} \u00b1 {DOP_err:.4f}")
print("=" * 55)

# ── Goodness of fit ───────────────────────────────────────────────────────
residuals = power_avg_corr - malus_law(angle_rad, *popt)
chi2 = np.sum((residuals / sigma_total) ** 2)
ndof = len(angle_deg) - len(popt)
print(f"  \u03c7\u00b2/ndf  = {chi2:.2f} / {ndof}  =  {chi2/ndof:.2f}")
print("=" * 55)

# ── Fit confidence band (1-σ) via linear error propagation ────────────────
theta_smooth = np.linspace(0, np.pi, 500)
power_smooth = malus_law(theta_smooth, *popt)

# Jacobian of malus_law w.r.t. (I_max, I_min, theta_0) at each smooth θ
def malus_jacobian(theta, I_max, I_min, theta_0):
    """Returns (N, 3) Jacobian matrix for the three fit parameters."""
    c  = np.cos(theta - theta_0)
    s  = np.sin(theta - theta_0)
    dP_dImax   = c ** 2
    dP_dImin   = np.ones_like(theta)
    dP_dtheta0 = 2 * I_max * c * s       # chain rule
    return np.column_stack([dP_dImax, dP_dImin, dP_dtheta0])

J = malus_jacobian(theta_smooth, *popt)          # (500, 3)
# σ²(P) at each θ = J · pcov · Jᵀ  (diagonal elements)
band_var = np.einsum("ij,jk,ik->i", J, pcov, J)
band_sigma = np.sqrt(band_var)                    # 1-σ band

# ── Figure 1: Data + Fit + Uncertainty Band ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# 1-σ confidence band
ax.fill_between(
    np.rad2deg(theta_smooth),
    power_smooth - band_sigma,
    power_smooth + band_sigma,
    color="crimson", alpha=0.15,
    label=r"$1\sigma$ confidence band",
)

# Data with error bars
ax.errorbar(
    angle_deg, power_avg_corr, yerr=sigma_total,
    fmt="o", capsize=3, color="royalblue", zorder=3,
    label="Data (background-subtracted)",
)

# Best-fit curve
ax.plot(
    np.rad2deg(theta_smooth), power_smooth,
    "-", color="crimson", linewidth=2, zorder=2,
    label=(
        r"$\cos^2$ fit:  "
        rf"$I_{{\max}} = {I_max_fit:.1f} \pm {I_max_err:.1f}$ $\mu$W,  "
        rf"$I_{{\min}} = {I_min_fit:.1f} \pm {I_min_err:.1f}$ $\mu$W"
    ),
)

ax.set_xlabel(r"Polarizer Angle ($^\circ$)", fontsize=13)
ax.set_ylabel(r"Power ($\mu$W)", fontsize=13)
ax.set_title(
    r"Malus's Law $\cos^2$ Fit  —  "
    rf"$\theta_0 = {np.rad2deg(theta0_fit):.1f} \pm {np.rad2deg(theta0_err):.1f}^\circ$,  "
    rf"$\chi^2_{{\nu}} = {chi2/ndof:.2f}$",
    fontsize=13,
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/malus_law_fit.png", dpi=200)
print("\nSaved  figures/malus_law_fit.png")

# ── Figure 2: Residuals with uncertainty ──────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.errorbar(
    angle_deg, residuals, yerr=sigma_total,
    fmt="o", capsize=3, color="royalblue",
)
ax2.axhline(0, color="crimson", linewidth=1.5)
ax2.set_xlabel(r"Polarizer Angle ($^\circ$)", fontsize=13)
ax2.set_ylabel(r"Residual ($\mu$W)", fontsize=13)
ax2.set_title(
    r"Fit Residuals  —  "
    rf"$\chi^2 / \mathrm{{ndf}} = {chi2:.1f}\,/\,{ndof} = {chi2/ndof:.2f}$",
    fontsize=13,
)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("figures/malus_law_residuals.png", dpi=200)
print("Saved  figures/malus_law_residuals.png")

plt.show()

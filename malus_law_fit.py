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

# ── Subtract background (0.95 µW) ──────────────────────────────────────────
background = 0.95

power_min_corr = power_min - background
power_max_corr = power_max - background
power_avg_corr = power_avg - background

# Uncertainty: half the spread between min and max readings
power_uncertainty = (power_max_corr - power_min_corr) / 2.0
# Set a minimum uncertainty floor so zero-spread points don't break the fit
power_uncertainty = np.maximum(power_uncertainty, 0.5)

# ── Malus's Law model ──────────────────────────────────────────────────────
# P(θ) = I_max * cos²(θ − θ_0) + I_min
def malus_law(theta, I_max, I_min, theta_0):
    return I_max * np.cos(theta - theta_0) ** 2 + I_min


# ── Fit ────────────────────────────────────────────────────────────────────
p0 = [340.0, 1.0, np.deg2rad(120)]  # initial guesses
popt, pcov = curve_fit(
    malus_law, angle_rad, power_avg_corr,
    p0=p0, sigma=power_uncertainty, absolute_sigma=True
)
perr = np.sqrt(np.diag(pcov))  # 1-σ parameter uncertainties

I_max_fit, I_min_fit, theta0_fit = popt
I_max_err, I_min_err, theta0_err = perr

print("=" * 55)
print("         Malus's Law Fit Results")
print("=" * 55)
print(f"  I_max   = {I_max_fit:8.2f} ± {I_max_err:.2f} µW")
print(f"  I_min   = {I_min_fit:8.2f} ± {I_min_err:.2f} µW")
print(f"  θ_0     = {np.rad2deg(theta0_fit):8.2f} ± {np.rad2deg(theta0_err):.2f}°")
print()

# ── Degree of polarization ─────────────────────────────────────────────────
# DOP = (I_max − I_min) / (I_max + I_min)   using fitted parameters
DOP = (I_max_fit - I_min_fit) / (I_max_fit + I_min_fit)

# Propagate uncertainty:  DOP = (a-b)/(a+b)
#   ∂DOP/∂a =  2b / (a+b)²
#   ∂DOP/∂b = −2a / (a+b)²
dDOP_dImax = 2 * I_min_fit / (I_max_fit + I_min_fit) ** 2
dDOP_dImin = -2 * I_max_fit / (I_max_fit + I_min_fit) ** 2
DOP_err = np.sqrt((dDOP_dImax * I_max_err) ** 2 + (dDOP_dImin * I_min_err) ** 2)

print(f"  Degree of polarization = {DOP:.4f} ± {DOP_err:.4f}")
print("=" * 55)

# ── Reduced chi-squared ───────────────────────────────────────────────────
residuals = power_avg_corr - malus_law(angle_rad, *popt)
chi2 = np.sum((residuals / power_uncertainty) ** 2)
ndof = len(angle_deg) - len(popt)
print(f"  χ²/ndf  = {chi2:.2f} / {ndof}  =  {chi2/ndof:.2f}")
print("=" * 55)

# ── Smooth curve for plotting ──────────────────────────────────────────────
theta_smooth = np.linspace(0, np.pi, 500)
power_smooth = malus_law(theta_smooth, *popt)

# ── Figure 1: Data + Fit ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    angle_deg, power_avg_corr, yerr=power_uncertainty,
    fmt="o", capsize=3, label="Data (background subtracted)", color="royalblue"
)
ax.plot(
    np.rad2deg(theta_smooth), power_smooth,
    "-", color="crimson", linewidth=2,
    label=(
        f"Cos^2 Fit: $I_{{max}}$={I_max_fit:.1f} µW, "
        f"$I_{{min}}$={I_min_fit:.1f} µW, "
        f"$\\theta_0$={np.rad2deg(theta0_fit):.1f}°"
    ),
)
ax.set_xlabel("Polarizer Angle (°)", fontsize=13)
ax.set_ylabel("Power (µW)", fontsize=13)
ax.set_title("Cos^2 Fit", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("malus_law_fit.png", dpi=200)
print("\nSaved  malus_law_fit.png")

# ── Figure 2: Residuals ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 2.5))
ax2.errorbar(
    angle_deg, residuals, yerr=power_uncertainty,
    fmt="o", capsize=3, color="royalblue"
)
ax2.axhline(0, color="crimson", linewidth=1.5)
ax2.set_xlabel("Polarizer Angle", fontsize=13)
ax2.set_ylabel("Residual (µW)", fontsize=13)
ax2.set_title("Residuals", fontsize=14)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("malus_law_residuals.png", dpi=200)
print("Saved  malus_law_residuals.png")

plt.show()

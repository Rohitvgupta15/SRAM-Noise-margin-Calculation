import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Step 1: Load CSV data ---
data = pd.read_csv("hold.csv")
x_lower = data.iloc[:, 0].values
y_lower = data.iloc[:, 1].values
x_upper = data.iloc[:, 2].values
y_upper = data.iloc[:, 3].values

# --- Step 2: Interpolate finely ---
interp_lower = interp1d(x_lower, y_lower, kind='cubic', fill_value='extrapolate')
interp_upper = interp1d(x_upper, y_upper, kind='cubic', fill_value='extrapolate')

x_fine = np.linspace(max(x_lower.min(), x_upper.min()),
                     min(x_lower.max(), x_upper.max()), 5000)
y_lower_f = interp_lower(x_fine)
y_upper_f = interp_upper(x_fine)

# --- Function to find max diagonal square ---
def find_max_square(x_fine, y_lower_f, y_upper_f, tol=1e-5):
    max_s = 0
    best_idx_lower = 0
    for i, y0 in enumerate(y_lower_f):
        x0 = x_fine[i]
        s_candidates = []
        for j, y1 in enumerate(y_upper_f):
            x1 = x_fine[j]
            s_x = x1 - x0
            s_y = y1 - y0
            if s_x > 0 and abs(s_x - s_y) < tol:
                s_candidates.append(s_x)
        if s_candidates:
            s_max_candidate = max(s_candidates)
            if s_max_candidate > max_s:
                max_s = s_max_candidate
                best_idx_lower = i
    x0_opt = x_fine[best_idx_lower]
    y0_opt = y_lower_f[best_idx_lower]
    return x0_opt, y0_opt, max_s

# --- Step 3: Upper SNM ---
x0_up, y0_up, s_up = find_max_square(x_fine, y_lower_f, y_upper_f)

# --- Step 4: Lower SNM (swap curves) ---
x0_low, y0_low, s_low = find_max_square(x_fine, y_upper_f, y_lower_f)

# --- Step 5: Square coordinates ---
def get_square_coords(x0, y0, s):
    square_x = [x0, x0 + s, x0 + s, x0, x0]
    square_y = [y0, y0, y0 + s, y0 + s, y0]
    diag_x = [x0, x0 + s]
    diag_y = [y0, y0 + s]
    diag_length = np.sqrt(2) * s
    return square_x, square_y, diag_x, diag_y, diag_length

sq_up = get_square_coords(x0_up, y0_up, s_up)
sq_low = get_square_coords(x0_low, y0_low, s_low)

# --- Step 6: Plot curves and squares ---
plt.figure(figsize=(8, 8))
plt.plot(x_lower, y_lower, 'r.', alpha=0.3, label='Lower Curve')
plt.plot(x_upper, y_upper, 'b.', alpha=0.3, label='Upper Curve')
plt.plot(x_fine, y_lower_f, 'r-', lw=2, label='Lower Curve Interp')
plt.plot(x_fine, y_upper_f, 'b-', lw=2, label='Upper Curve Interp')

# Plot upper SNM square (only if fitted)
if s_up > 0:
    plt.plot(sq_up[0], sq_up[1], 'k--', lw=2, label='Upper Max Square')
    plt.plot(sq_up[2], sq_up[3], 'g-', lw=2, label=f'Upper Diagonal = {sq_up[4]:.4f} V')
    plt.scatter([x0_up, x0_up + s_up], [y0_up, y0_up + s_up], color='k', zorder=5)

# Plot lower SNM square (only if fitted)
if s_low > 0:
    plt.plot(sq_low[0], sq_low[1], 'm--', lw=2, label='Lower Max Square')
    plt.plot(sq_low[2], sq_low[3], 'c-', lw=2, label=f'Lower Diagonal = {sq_low[4]:.4f} V')
    plt.scatter([x0_low, x0_low + s_low], [y0_low, y0_low + s_low], color='m', zorder=5)

plt.title("SRAM Butterfly Curve with Maximum Diagonal Squares")
plt.xlabel("Q / Qb X")
plt.ylabel("Q / Qb Y")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# --- Step 7: SNM Calculation Logic ---
if s_up > 0 and s_low > 0:
    # Both squares fitted
    SNM = min(s_up, s_low)
    case = "Both squares fitted"
elif s_up > 0:
    # Only upper square fitted
    SNM = s_up
    case = "Only upper square fitted"
elif s_low > 0:
    # Only lower square fitted
    SNM = s_low
    case = "Only lower square fitted"
else:
    SNM = 0
    case = "No valid square found"

diag_SNM = np.sqrt(2) * SNM

# --- Step 8: Print Results ---
print("==== SRAM SNM Results ====")
print(f"Case: {case}")
print(f"Upper SNM square side = {s_up:.4f} V")
print(f"Lower SNM square side = {s_low:.4f} V")
print(f"Static Noise Margin (SNM) = {SNM:.4f} V")
print(f"Diagonal of SNM square = {diag_SNM:.4f} V")

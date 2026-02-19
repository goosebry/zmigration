"""
Refine Model 6 coefficients using all 7 known actual scores
"""

import numpy as np

# All known data points with actual scores
data = [
    # (Structure, Hero, Tech, Truck, Actual)
    (82.69843, 97.15876, 36.84337, 22.74237, 121),      # Mibu Wolf
    (60.92832, 83.5066, 28.75272, 20.01968, 91),        # Guidingstar
    (51.48694, 53.54554, 17.55005, 8.835235, 33.8),     # Shawnic
    (42.68917, 51.04704, 13.65734, 7.650276, 31),       # Swade Toobe
    (85.22689, 74.70785, 28.47319, 12.14937, 69),       # Steel Fury
    (63.637343, 65.886325, 22.082430, 13.665822, 54.7),   # Bon/Axel (precise)
    (59, 53, 15, 9.5, 30.5),                             # Krys
]

X = np.array([[d[0], d[1], d[2], d[3]] for d in data])
y_actual = np.array([d[4] for d in data])
totals = X.sum(axis=1)

print("=" * 70)
print("REFINING MODEL WITH 7 KNOWN DATA POINTS")
print("=" * 70)

print("\nData summary:")
print(f"{'Name':<15} {'Total':>8} {'Actual':>8}")
print("-" * 35)
names = ['Mibu Wolf', 'Guidingstar', 'Shawnic', 'Swade Toobe', 'Steel Fury', 'Bon/Axel', 'Krys']
for i, name in enumerate(names):
    print(f"{name:<15} {totals[i]:>8.1f} {y_actual[i]:>8.1f}")

# Current Model 6: Base = 0.2S + 0.7H + 0.1T + 0.3V, Score = 0.0092 * Base^2.0886
print("\n" + "=" * 70)
print("CURRENT MODEL 6 PERFORMANCE")
print("=" * 70)

def current_model6(S, H, T, V):
    base = 0.200 * S + 0.700 * H + 0.100 * T + 0.300 * V
    return 0.0092 * (base ** 2.0886)

current_preds = np.array([current_model6(*row) for row in X])
current_errors = current_preds - y_actual
print(f"\nCurrent coefficients: 0.2S + 0.7H + 0.1T + 0.3V, mult=0.0092, exp=2.0886")
print(f"Mean Absolute Error: {np.mean(np.abs(current_errors)):.2f}")
print(f"Max Error: {np.max(np.abs(current_errors)):.2f}")

# Grid search for better coefficients
print("\n" + "=" * 70)
print("SEARCHING FOR OPTIMAL COEFFICIENTS")
print("=" * 70)

best_mae = float('inf')
best_params = None

# Search ranges based on current values
for a in np.linspace(0.10, 0.35, 10):  # Structure weight
    for b in np.linspace(0.50, 0.90, 10):  # Hero weight
        for c in np.linspace(0.05, 0.25, 8):  # Tech weight
            for d in np.linspace(0.15, 0.45, 8):  # Truck weight
                for mult in np.linspace(0.005, 0.015, 10):
                    for exp in np.linspace(1.8, 2.3, 10):
                        bases = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d * X[:, 3]
                        preds = mult * np.power(bases, exp)
                        mae = np.mean(np.abs(preds - y_actual))
                        
                        if mae < best_mae:
                            best_mae = mae
                            best_params = (a, b, c, d, mult, exp)

a, b, c, d, mult, exp = best_params
print(f"\nBest parameters found:")
print(f"  Structure weight: {a:.4f}")
print(f"  Hero weight:      {b:.4f}")
print(f"  Tech weight:      {c:.4f}")
print(f"  Truck weight:     {d:.4f}")
print(f"  Multiplier:       {mult:.6f}")
print(f"  Exponent:         {exp:.4f}")

print(f"\nFormula: Score = {mult:.6f} × ({a:.3f}S + {b:.3f}H + {c:.3f}T + {d:.3f}V)^{exp:.4f}")

# Test new model
print("\n" + "=" * 70)
print("NEW MODEL PERFORMANCE")
print("=" * 70)

def new_model(S, H, T, V):
    base = a * S + b * H + c * T + d * V
    return mult * (base ** exp)

new_preds = np.array([new_model(*row) for row in X])
new_errors = new_preds - y_actual

print(f"\n{'Name':<15} {'Actual':>8} {'Old Pred':>10} {'New Pred':>10} {'Old Err':>10} {'New Err':>10}")
print("-" * 70)
for i, name in enumerate(names):
    print(f"{name:<15} {y_actual[i]:>8.1f} {current_preds[i]:>10.1f} {new_preds[i]:>10.1f} {current_errors[i]:>+10.1f} {new_errors[i]:>+10.1f}")

print("-" * 70)
print(f"{'Mean Abs Error':<15} {'':<8} {np.mean(np.abs(current_errors)):>10.2f} {np.mean(np.abs(new_errors)):>10.2f}")
print(f"{'Max Error':<15} {'':<8} {np.max(np.abs(current_errors)):>10.2f} {np.max(np.abs(new_errors)):>10.2f}")

print("\n" + "=" * 70)
print("JAVASCRIPT UPDATE")
print("=" * 70)
print(f"""
// NEW Model 6 coefficients (refined with 7 data points)
const base6 = {a:.4f} * structure + {b:.4f} * hero + {c:.4f} * tech + {d:.4f} * truck;
const model6 = {mult:.6f} * Math.pow(base6, {exp:.4f});
""")

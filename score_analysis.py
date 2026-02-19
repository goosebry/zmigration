"""
Z Shooter Run Season 2 Migration Score Reverse Engineering

Hypothesis: Base calculation with coefficient scaling
Base = a*Structure + b*Hero + c*Tech + d*Truck + intercept
Final = Base * coefficient (or some scaling function)
"""

import numpy as np
import math

# Data from screenshot (values in millions)
data = {
    'Mibu Wolf':    {'Structure': 82.69843, 'Hero': 97.15876, 'Tech': 36.84337, 'Truck': 22.74237, 'Est': 65, 'Actual': 121},
    'Guidingstar':  {'Structure': 60.92832, 'Hero': 83.5066,  'Tech': 28.75272, 'Truck': 20.01968, 'Est': 50, 'Actual': 91},
    'Shawnic':      {'Structure': 51.48694, 'Hero': 53.54554, 'Tech': 17.55005, 'Truck': 8.835235, 'Est': 34, 'Actual': 33.8},
    'Hermes':       {'Structure': 65.45359, 'Hero': 63.64449, 'Tech': 20.47752, 'Truck': 11.49897, 'Est': 43, 'Actual': None},
    'Swade Toobe':  {'Structure': 42.68917, 'Hero': 51.04704, 'Tech': 13.65734, 'Truck': 7.650276, 'Est': 28, 'Actual': 31},
}

# Extract data with known actual scores
names_with_actual = [k for k, v in data.items() if v['Actual'] is not None]
X = np.array([[data[n]['Structure'], data[n]['Hero'], data[n]['Tech'], data[n]['Truck']] for n in names_with_actual])
y_actual = np.array([data[n]['Actual'] for n in names_with_actual])
y_est = np.array([data[n]['Est'] for n in names_with_actual])

print("=" * 60)
print("Z SHOOTER RUN - SEASON 2 MIGRATION SCORE ANALYSIS")
print("=" * 60)

print("\n--- DATA SUMMARY ---")
for name in names_with_actual:
    d = data[name]
    total = d['Structure'] + d['Hero'] + d['Tech'] + d['Truck']
    print(f"{name}: Total={total:.2f}M, Est={d['Est']}, Actual={d['Actual']}, Ratio={d['Actual']/d['Est']:.3f}")

# ============================================================
# MODEL 1: Simple Linear Regression
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: Simple Linear (Base = aS + bH + cT + dV + intercept)")
print("=" * 60)

# Add intercept column
X_with_intercept = np.column_stack([X, np.ones(len(X))])

# Solve using least squares
coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y_actual, rcond=None)
a, b, c, d, intercept = coeffs

print(f"\nLinear coefficients:")
print(f"  Structure:  {a:.6f}")
print(f"  Hero:       {b:.6f}")
print(f"  Tech:       {c:.6f}")
print(f"  Truck:      {d:.6f}")
print(f"  Intercept:  {intercept:.6f}")

print(f"\nFormula: Score = {a:.4f}*S + {b:.4f}*H + {c:.4f}*T + {d:.4f}*V + ({intercept:.4f})")

# Predictions
print("\nPredictions vs Actual:")
for i, name in enumerate(names_with_actual):
    pred = np.dot(X_with_intercept[i], coeffs)
    actual = y_actual[i]
    error = pred - actual
    print(f"  {name}: Predicted={pred:.2f}, Actual={actual}, Error={error:+.2f}")

# ============================================================
# MODEL 2: Two-stage (Base * Coefficient)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: Two-Stage (Base calculation, then scaling)")
print("=" * 60)

# First, let's see if Est. Ranking Power follows a pattern
print("\nAnalyzing Est. Ranking Power vs inputs:")
X_est = np.column_stack([X, np.ones(len(X))])
coeffs_est, _, _, _ = np.linalg.lstsq(X_est, y_est, rcond=None)
print(f"  Est = {coeffs_est[0]:.4f}*S + {coeffs_est[1]:.4f}*H + {coeffs_est[2]:.4f}*T + {coeffs_est[3]:.4f}*V + ({coeffs_est[4]:.4f})")

# Now analyze the scaling from Est to Actual
print("\nScaling factors (Actual / Est):")
scaling_factors = y_actual / y_est
for i, name in enumerate(names_with_actual):
    total_power = X[i].sum()
    print(f"  {name}: Scale={scaling_factors[i]:.4f}, Total Power={total_power:.2f}M")

# ============================================================
# MODEL 3: Non-linear scaling based on total power
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Non-linear scaling (larger accounts scale up)")
print("=" * 60)

total_powers = X.sum(axis=1)

# Analyze the ratio pattern
print("\nAnalyzing Actual/Est ratio vs Total Power:")
for i, name in enumerate(names_with_actual):
    ratio = y_actual[i] / y_est[i]
    total = total_powers[i]
    log_total = np.log(total)
    print(f"  {name}: Ratio={ratio:.3f}, Total={total:.2f}M, ln(Total)={log_total:.3f}")

# Try simple scaling: Ratio ≈ a * ln(Total) + b
log_totals = np.log(total_powers)
ratios = y_actual / y_est

# Fit linear: ratio = a * ln(total) + b
A_ratio = np.column_stack([log_totals, np.ones(len(log_totals))])
ratio_coeffs, _, _, _ = np.linalg.lstsq(A_ratio, ratios, rcond=None)
a_ratio, b_ratio = ratio_coeffs

print(f"\nRatio formula: Actual/Est = {a_ratio:.4f} * ln(TotalPower) + ({b_ratio:.4f})")
print("Therefore: Actual = Est * [{:.4f} * ln(TotalPower) + ({:.4f})]".format(a_ratio, b_ratio))

print("\nPredictions:")
for i, name in enumerate(names_with_actual):
    pred_ratio = a_ratio * log_totals[i] + b_ratio
    pred = y_est[i] * pred_ratio
    print(f"  {name}: Predicted={pred:.2f}, Actual={y_actual[i]}, Error={pred-y_actual[i]:+.2f}")

# ============================================================
# MODEL 4: Your suggested base formula structure
# ============================================================
print("\n" + "=" * 60)
print("MODEL 4: Base + Compression/Scaling coefficient")
print("=" * 60)

# Based on your suggestion: Base = 0.48S + 0.62H + 0.18T + 0.27V - 14.9
# Then apply a scaling coefficient

# Test your suggested formula
def calc_base(row, a=0.48, b=0.62, c=0.18, d=0.27, intercept=-14.9):
    return a * row[0] + b * row[1] + c * row[2] + d * row[3] + intercept

print("\nTesting your suggested base formula:")
print("Base = 0.48*S + 0.62*H + 0.18*T + 0.27*V - 14.9")
print()

for i, name in enumerate(names_with_actual):
    base = calc_base(X[i])
    actual = y_actual[i]
    if base > 0:
        coef_needed = actual / base
    else:
        coef_needed = float('nan')
    print(f"  {name}: Base={base:.2f}, Actual={actual}, Coefficient needed={coef_needed:.4f}")

# Now let's find optimal base coefficients
print("\n\nFitting optimal base formula coefficients:")

# Try: Actual = (aS + bH + cT + dV + int) * scaling_factor(total)
# Where scaling_factor = c1 * ln(total) + c2

def grid_search_model():
    best_error = float('inf')
    best_params = None
    
    # Grid search over reasonable ranges
    for a in np.linspace(0.3, 0.7, 5):
        for b in np.linspace(0.4, 0.8, 5):
            for c in np.linspace(0.1, 0.4, 5):
                for d in np.linspace(0.1, 0.5, 5):
                    for intercept in np.linspace(-30, 0, 5):
                        for c1 in np.linspace(0.2, 0.6, 5):
                            for c2 in np.linspace(-0.5, 0.5, 5):
                                base = a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d * X[:, 3] + intercept
                                scaling = c1 * np.log(total_powers) + c2
                                pred = base * scaling
                                error = np.sum((pred - y_actual) ** 2)
                                if error < best_error:
                                    best_error = error
                                    best_params = (a, b, c, d, intercept, c1, c2)
    return best_params, best_error

# Simplified approach: use linear regression for base, then fit scaling
print("\nSimplified two-stage approach:")
print("Stage 1: Fit base formula to get reasonable coefficients")
print("Stage 2: Determine scaling factor pattern")

# For the scaling, we already found: ratio = a_ratio * ln(total) + b_ratio
# So: Actual = Est * (a_ratio * ln(total) + b_ratio)
# And Est seems to follow a pattern too

# Let's verify the Est formula
print("\n\nVerifying Est. Ranking Power formula:")
for i, name in enumerate(names_with_actual):
    d_item = data[name]
    total = X[i].sum()
    # Simple approximation: Est ≈ total / some_factor
    ratio_to_total = d_item['Est'] / total
    print(f"  {name}: Est={d_item['Est']}, Total={total:.2f}M, Est/Total={ratio_to_total:.4f}")

# ============================================================
# MODEL 5: Combined formula with power-law scaling
# ============================================================
print("\n" + "=" * 60)
print("MODEL 5: Combined formula with power-law scaling")
print("=" * 60)

# Hypothesis: Score = Base^exponent * multiplier
# Or: Score = multiplier * Total^exponent + offset

# Let's try: Score = a * Total^b + c
def fit_power_law():
    # Using log-linear regression: ln(Actual) = ln(a) + b*ln(Total)
    # Simplified: Actual ≈ a * Total^b
    
    ln_actual = np.log(y_actual)
    ln_total = np.log(total_powers)
    
    A = np.column_stack([ln_total, np.ones(len(ln_total))])
    coeffs_power, _, _, _ = np.linalg.lstsq(A, ln_actual, rcond=None)
    
    b_exp = coeffs_power[0]
    a_mult = np.exp(coeffs_power[1])
    
    return a_mult, b_exp

a_power, b_power = fit_power_law()
print(f"\nPower-law model: Score = {a_power:.4f} * TotalPower^{b_power:.4f}")

print("\nPredictions:")
for i, name in enumerate(names_with_actual):
    pred = a_power * (total_powers[i] ** b_power)
    print(f"  {name}: Predicted={pred:.2f}, Actual={y_actual[i]}, Error={pred-y_actual[i]:+.2f}")

# ============================================================
# MODEL 6: Weighted components with non-linear scaling
# ============================================================
print("\n" + "=" * 60)
print("MODEL 6: Finding optimal component weights")
print("=" * 60)

# Use iterative refinement to find best weights
def evaluate_model(weights, data_X, data_y):
    a, b, c, d = weights[:4]
    base = a * data_X[:, 0] + b * data_X[:, 1] + c * data_X[:, 2] + d * data_X[:, 3]
    
    # Fit the remaining transformation
    A = np.column_stack([np.log(base), np.ones(len(base))])
    transform_coeffs, _, _, _ = np.linalg.lstsq(A, np.log(data_y), rcond=None)
    
    pred = np.exp(transform_coeffs[1]) * np.power(base, transform_coeffs[0])
    error = np.sum((pred - data_y) ** 2)
    return error, transform_coeffs

# Simple grid search for component weights (relative importance)
print("\nSearching for optimal component weights...")
best_error = float('inf')
best_weights = None
best_transform = None

for a in np.linspace(0.2, 0.6, 5):
    for b in np.linspace(0.3, 0.7, 5):
        for c in np.linspace(0.1, 0.4, 5):
            for d in np.linspace(0.05, 0.3, 5):
                weights = [a, b, c, d]
                try:
                    error, transform = evaluate_model(weights, X, y_actual)
                    if error < best_error:
                        best_error = error
                        best_weights = weights
                        best_transform = transform
                except:
                    pass

if best_weights:
    a, b, c, d = best_weights
    print(f"\nBest weights found:")
    print(f"  Structure: {a:.3f}")
    print(f"  Hero:      {b:.3f}")
    print(f"  Tech:      {c:.3f}")
    print(f"  Truck:     {d:.3f}")
    
    exponent, log_mult = best_transform
    mult = np.exp(log_mult)
    print(f"\nTransformation: Score = {mult:.4f} * Base^{exponent:.4f}")
    print(f"Where Base = {a:.3f}*S + {b:.3f}*H + {c:.3f}*T + {d:.3f}*V")
    
    print("\nPredictions:")
    for i, name in enumerate(names_with_actual):
        base = a * X[i, 0] + b * X[i, 1] + c * X[i, 2] + d * X[i, 3]
        pred = mult * (base ** exponent)
        print(f"  {name}: Base={base:.2f}, Predicted={pred:.2f}, Actual={y_actual[i]}, Error={pred-y_actual[i]:+.2f}")

# ============================================================
# PREDICT HERMES (unknown actual)
# ============================================================
print("\n" + "=" * 60)
print("PREDICTIONS FOR HERMES (unknown actual)")
print("=" * 60)

hermes_data = np.array([[data['Hermes']['Structure'], data['Hermes']['Hero'], 
                         data['Hermes']['Tech'], data['Hermes']['Truck']]])

print(f"Input: S={hermes_data[0,0]:.2f}, H={hermes_data[0,1]:.2f}, T={hermes_data[0,2]:.2f}, V={hermes_data[0,3]:.2f}")
print(f"Est. Ranking Power: {data['Hermes']['Est']}")

# Model 1 prediction
hermes_X = np.column_stack([hermes_data, np.ones(1)])
pred1 = np.dot(hermes_X, coeffs)[0]
print(f"\nModel 1 (Linear): {pred1:.2f}")

# Model 3 prediction (using ratio formula)
hermes_total = hermes_data.sum()
hermes_ratio = a_ratio * np.log(hermes_total) + b_ratio
pred3 = data['Hermes']['Est'] * hermes_ratio
print(f"Model 3 (Est * scaling): {pred3:.2f}")

# Model 5 prediction (power law)
pred5 = a_power * (hermes_total ** b_power)
print(f"Model 5 (Power law): {pred5:.2f}")

# Model 6 prediction
if best_weights:
    hermes_base = a * hermes_data[0, 0] + b * hermes_data[0, 1] + c * hermes_data[0, 2] + d * hermes_data[0, 3]
    pred6 = mult * (hermes_base ** exponent)
    print(f"Model 6 (Weighted + transform): {pred6:.2f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key observations:
1. Larger accounts (Mibu Wolf, Guidingstar) have Actual >> Est
2. Smaller accounts (Shawnic, Swade Toobe) have Actual ≈ Est
3. This suggests a non-linear scaling that amplifies larger accounts

The scaling appears to follow a pattern where:
- Accounts with higher total power get a multiplier > 1
- Accounts with lower total power get a multiplier ≈ 1 or slightly less

This could be intentional game design to reward progression.
""")

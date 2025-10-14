import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

# -------------------------------
# 1) Locate Excel file
# -------------------------------
# Use the specific file path for the EJSL dashboard
file_path = "/Users/brianbutler/Desktop/EJSL/Dashboard EJS Draft.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Cannot find Excel file at {file_path}")

xl = pd.ExcelFile(file_path, engine="openpyxl")
print("Available sheets:", xl.sheet_names)

# -------------------------------
# 2) Parse weekly baseline % table
# -------------------------------
weekly_df = xl.parse("Total Client % with All", header=0)
weekly_df.columns = weekly_df.columns.str.strip()

keep = ["Week (2025)", "PJ2H %", "PORT %", "BHOP %"]

for col in keep:
    if col not in weekly_df.columns:
        raise KeyError(f"Expected column '{col}' not found in weekly data")

weekly_df = weekly_df.dropna(subset=keep)
weekly_df["Week (2025)"] = weekly_df["Week (2025)"].astype(str)
weekly_df[keep[1:]] = weekly_df[keep[1:]].astype(float)

# Calculate weekly changes (absolute number change)
weekly_df["PJ2H Δ%"] = weekly_df["PJ2H %"].diff()
weekly_df["PORT Δ%"] = weekly_df["PORT %"].diff()
weekly_df["BHOP Δ%"] = weekly_df["BHOP %"].diff()

print("\n=== WEEK-OVER-WEEK BASELINE % CHANGES ===")
print(weekly_df[["Week (2025)", "PJ2H %", "PJ2H Δ%", "PORT %", "PORT Δ%", "BHOP %", "BHOP Δ%"]].tail(10))

# Calculate average weekly change
print("\n=== AVERAGE WEEKLY CHANGE ===")
print(f"PJ2H: {weekly_df['PJ2H Δ%'].mean():.2f}% per week")
print(f"PORT: {weekly_df['PORT Δ%'].mean():.2f}% per week")
print(f"BHOP: {weekly_df['BHOP Δ%'].mean():.2f}% per week")

# -------------------------------
# 3) Project baselines to November
# -------------------------------
weeks = np.arange(len(weekly_df))

# Determine how many weeks to project (to end of November 2025)
# Assume current data ends around mid-October, project ~6 weeks to end of Nov
future_weeks = np.arange(len(weekly_df), len(weekly_df) + 6)


def project_trend(series, weeks_idx, future_idx):
    lr = LinearRegression()
    lr.fit(weeks_idx.reshape(-1, 1), series.values)
    all_idx = np.concatenate([weeks_idx, future_idx])
    predictions = lr.predict(all_idx.reshape(-1, 1))

    # Calculate weekly growth rate
    slope = lr.coef_[0]

    return all_idx, predictions, slope


pj2h_idx, pj2h_proj, pj2h_slope = project_trend(weekly_df["PJ2H %"], weeks, future_weeks)
port_idx, port_proj, port_slope = project_trend(weekly_df["PORT %"], weeks, future_weeks)
bhop_idx, bhop_proj, bhop_slope = project_trend(weekly_df["BHOP %"], weeks, future_weeks)

print(f"\n=== BASELINE % GROWTH RATES (per week) ===")
print(f"PJ2H: {pj2h_slope:.3f}% per week")
print(f"PORT: {port_slope:.3f}% per week")
print(f"BHOP: {bhop_slope:.3f}% per week")

# -------------------------------
# 4) Load outcomes data
# -------------------------------
# Look for sheets that might contain outcomes data
prog_sheets = [s for s in xl.sheet_names if "Progress" in s]

print(f"\n=== FOUND PROGRESS SHEETS ===")
for sheet in prog_sheets:
    print(f"  - {sheet}")

if len(prog_sheets) == 0:
    print("\nNo 'Progress' sheets found. Looking for alternative outcome data...")
    # Look for any sheets that might have outcome data
    prog_sheets = [s for s in xl.sheet_names if any(keyword in s.lower() for keyword in ['outcome', 'result', 'data'])]

if len(prog_sheets) < 1:
    print("\nWarning: No outcome sheets found. Skipping outcomes analysis.")
    skip_outcomes = True
else:
    skip_outcomes = False
    # Use the last sheet if only one exists, otherwise use last two
    if len(prog_sheets) == 1:
        sheet_prev, sheet_latest = prog_sheets[0], prog_sheets[0]
        print(f"\nUsing single sheet '{sheet_latest}' for outcomes")
    else:
        sheet_prev, sheet_latest = prog_sheets[-2], prog_sheets[-1]
        print(f"\n=== OUTCOMES COMPARISON ===")
        print(f"Using '{sheet_prev}' vs '{sheet_latest}'")


def load_outcomes(sheet_name):
    # Read more rows to ensure we capture the data
    df = xl.parse(sheet_name, nrows=50)
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric_values = df_numeric.values.flatten()
    numeric_values = numeric_values[~np.isnan(numeric_values)]

    # Filter to reasonable outcome values (typically between 1 and 500)
    numeric_values = numeric_values[(numeric_values > 0) & (numeric_values < 1000)]

    if len(numeric_values) < 2:
        print(f"Warning: Sheet '{sheet_name}' found {len(numeric_values)} numeric values")
        print(f"Available numeric values: {numeric_values[:10]}")
        # Return default values if not found
        return 0, 0

    # Take first two reasonable values as short and long outcomes
    short, long = numeric_values[0], numeric_values[1]
    return short, long


if not skip_outcomes:
    short_prev, long_prev = load_outcomes(sheet_prev)
    short_latest, long_latest = load_outcomes(sheet_latest)

    print(f"Short-term outcomes: {short_prev} → {short_latest} (Δ: +{short_latest - short_prev})")
    print(f"Long-term outcomes: {long_prev} → {long_latest} (Δ: +{long_latest - long_prev})")
else:
    short_prev, long_prev = 0, 0
    short_latest, long_latest = 0, 0
    print("Skipping outcomes analysis - no data found")

# -------------------------------
# 5) Plot BASELINE % trends with projection to November
# -------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

# Plot actual data points
ax.scatter(weeks, weekly_df["PJ2H %"], c="blue", s=50, alpha=0.6, label="PJ2H % (actual)")
ax.scatter(weeks, weekly_df["PORT %"], c="green", s=50, alpha=0.6, label="PORT % (actual)")
ax.scatter(weeks, weekly_df["BHOP %"], c="red", s=50, alpha=0.6, label="BHOP % (actual)")

# Plot projection lines
split_point = len(weeks)
ax.plot(pj2h_idx[:split_point], pj2h_proj[:split_point], "-", c="blue", linewidth=2)
ax.plot(pj2h_idx[split_point - 1:], pj2h_proj[split_point - 1:], "--", c="blue", linewidth=2, alpha=0.7,
        label=f"PJ2H % proj ({pj2h_slope:.2f}%/wk)")

ax.plot(port_idx[:split_point], port_proj[:split_point], "-", c="green", linewidth=2)
ax.plot(port_idx[split_point - 1:], port_proj[split_point - 1:], "--", c="green", linewidth=2, alpha=0.7,
        label=f"PORT % proj ({port_slope:.2f}%/wk)")

ax.plot(bhop_idx[:split_point], bhop_proj[:split_point], "-", c="red", linewidth=2)
ax.plot(bhop_idx[split_point - 1:], bhop_proj[split_point - 1:], "--", c="red", linewidth=2, alpha=0.7,
        label=f"BHOP % proj ({bhop_slope:.2f}%/wk)")

# Add vertical line at projection start
ax.axvline(x=split_point - 0.5, color='gray', linestyle=':', alpha=0.5, label='Projection starts')

ax.set_title("Baseline Completion % by Program - Trend & Projection to November 2025", fontsize=14, fontweight='bold')
ax.set_xlabel("Week Index", fontsize=12)
ax.set_ylabel("% of Clients with Baselines Completed", fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/brianbutler/Desktop/EJSL/baselines_projection_to_november.png", dpi=300)
plt.close()

# -------------------------------
# 6) Plot OUTCOMES trends with projection to November
# -------------------------------
if not skip_outcomes and (short_latest > 0 or long_latest > 0):
    # Create time points for outcomes (2 actual, project 6 more)
    out_idx = np.array([0, 1])
    future_out = np.arange(2, 8)  # Project 6 more periods
    all_out_idx = np.concatenate([out_idx, future_out])

    # Fit linear models
    lr_short = LinearRegression().fit(out_idx.reshape(-1, 1), [short_prev, short_latest])
    lr_long = LinearRegression().fit(out_idx.reshape(-1, 1), [long_prev, long_latest])

    # Project
    short_proj = lr_short.predict(all_out_idx.reshape(-1, 1))
    long_proj = lr_long.predict(all_out_idx.reshape(-1, 1))

    # Calculate growth rates
    short_growth = lr_short.coef_[0]
    long_growth = lr_long.coef_[0]

    print(f"\n=== OUTCOMES GROWTH RATES (per period) ===")
    print(f"Short-term: +{short_growth:.2f} completions per period")
    print(f"Long-term: +{long_growth:.2f} completions per period")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot actual data
    ax.scatter(out_idx, [short_prev, short_latest], c="purple", s=100, zorder=3, label="Short-term (actual)")
    ax.scatter(out_idx, [long_prev, long_latest], c="orange", s=100, zorder=3, label="Long-term (actual)")

    # Plot trend lines and projections
    ax.plot(all_out_idx[:2], short_proj[:2], "-", c="purple", linewidth=2)
    ax.plot(all_out_idx[1:], short_proj[1:], "--", c="purple", linewidth=2, alpha=0.7,
            label=f"Short-term proj (+{short_growth:.1f}/period)")

    ax.plot(all_out_idx[:2], long_proj[:2], "-", c="orange", linewidth=2)
    ax.plot(all_out_idx[1:], long_proj[1:], "--", c="orange", linewidth=2, alpha=0.7,
            label=f"Long-term proj (+{long_growth:.1f}/period)")

    # Add vertical line
    ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5, label='Projection starts')

    # Format x-axis
    xticks_labels = [sheet_prev.split()[-1] if sheet_prev else "Prev",
                     sheet_latest.split()[-1] if sheet_latest else "Latest"] + [f"+{i}" for i in range(1, 7)]
    ax.set_xticks(all_out_idx)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')

    ax.set_title("Outcome Completions - Trend & Projection to November 2025", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Number of Outcomes Completed", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/brianbutler/Desktop/EJSL/outcomes_projection_to_november.png", dpi=300)
    plt.close()

    outcomes_saved = True

    # Store projections for summary
    short_proj_end = short_proj[-1]
    long_proj_end = long_proj[-1]
else:
    print("\nSkipping outcomes chart - insufficient data")
    outcomes_saved = False
    short_proj_end = 0
    long_proj_end = 0

# -------------------------------
# 7) Summary statistics table
# -------------------------------
print("\n" + "=" * 60)
print("SUMMARY: WEEKLY GROWTH ANALYSIS")
print("=" * 60)

summary_data = {
    "Program": ["PJ2H", "PORT", "BHOP"],
    "Current %": [weekly_df["PJ2H %"].iloc[-1], weekly_df["PORT %"].iloc[-1], weekly_df["BHOP %"].iloc[-1]],
    "Avg Weekly Δ": [weekly_df["PJ2H Δ%"].mean(), weekly_df["PORT Δ%"].mean(), weekly_df["BHOP Δ%"].mean()],
    "Trend (slope)": [pj2h_slope, port_slope, bhop_slope],
    "Projected Nov %": [pj2h_proj[-1], port_proj[-1], bhop_proj[-1]]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("OUTCOMES GROWTH ANALYSIS")
print("=" * 60)
if not skip_outcomes and outcomes_saved:
    print(f"Short-term: {short_prev} → {short_latest} → projected {short_proj_end:.0f} by November")
    print(f"Long-term:  {long_prev} → {long_latest} → projected {long_proj_end:.0f} by November")
else:
    print("No outcomes data available for analysis")

print("\n✓ Analysis complete! Charts saved to:")
print("  - /Users/brianbutler/Desktop/EJSL/baselines_projection_to_november.png")
if outcomes_saved:
    print("  - /Users/brianbutler/Desktop/EJSL/outcomes_projection_to_november.png")
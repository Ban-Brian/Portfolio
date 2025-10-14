import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# -------------------------------
# Configuration
# -------------------------------
FILEPATH = '/Users/brianbutler/Desktop/EJSL/Dashboard EJS Draft.xlsx'
OUTPUT_DIR = '/Users/brianbutler/Desktop/EJSL/'

print("=" * 80)
print("EJSL DASHBOARD ANALYSIS - TIDY FORMAT")
print("=" * 80)

xls = pd.ExcelFile(FILEPATH)
print(f"\nAvailable sheets: {xls.sheet_names}\n")


# -------------------------------
# Helper Functions
# -------------------------------
def parse_week_start(val):
    """Parse week ranges like 'April 13, 2025 to April 19, 2025' to week start date"""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if ' to ' in s:
        s = s.split(' to ')[0]
    try:
        return pd.to_datetime(s, errors='coerce')
    except:
        return pd.NaT


# -------------------------------
# 1) Fix Headers - Notes of All the Numbers
# -------------------------------
print("=" * 80)
print("STEP 1: EXTRACTING & TIDYING WEEKLY BASELINES")
print("=" * 80)

df_raw = pd.read_excel(FILEPATH, sheet_name='Notes of All the Numbers', header=0)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns found: {list(df_raw.columns)}\n")

# Find PORT columns by searching for keywords
port_week_col = None
port_active_col = None
port_baseline_col = None
port_percent_col = None

pj2h_week_col = None
pj2h_cols = []

for i, col in enumerate(df_raw.columns):
    col_str = str(col).strip()
    col_upper = col_str.upper()

    # PORT columns
    if 'PORT' in col_upper and 'WEEK' in col_upper and port_week_col is None:
        port_week_col = col
        print(f"  Found PORT week column: '{col}'")
    elif port_week_col and port_active_col is None and i > 0:
        # Next column after PORT week is likely Current Active
        port_active_col = col
        print(f"  Found PORT active column: '{col}'")
    elif port_active_col and port_baseline_col is None:
        # Next is Clients With Baselines
        port_baseline_col = col
        print(f"  Found PORT baseline column: '{col}'")
    elif port_baseline_col and port_percent_col is None:
        # Next is Percent
        port_percent_col = col
        print(f"  Found PORT percent column: '{col}'")

    # PJ2H columns
    if ('PJ2H' in col_upper or 'PS2H' in col_upper or 'PPJ2H' in col_upper) and 'WEEK' in col_upper:
        pj2h_week_col = col
        print(f"  Found PJ2H week column: '{col}'")
        # Next 3 columns are the PJ2H data
        if i + 3 < len(df_raw.columns):
            pj2h_cols = [df_raw.columns[i], df_raw.columns[i + 1],
                         df_raw.columns[i + 2], df_raw.columns[i + 3]]
            print(f"  PJ2H block columns: {pj2h_cols}")

# Build PORT dataframe
if all([port_week_col, port_active_col, port_baseline_col, port_percent_col]):
    port_df = df_raw[[port_week_col, port_active_col, port_baseline_col, port_percent_col]].copy()
    port_df.columns = ['Week', 'Current_Active', 'Clients_With_Baselines', 'Percent']
    port_df['Program'] = 'PORT'
    print("\n✓ PORT data extracted")
else:
    print("\n⚠ Could not find all PORT columns")
    port_df = pd.DataFrame()

# Build PJ2H dataframe
if len(pj2h_cols) == 4:
    pj2h_df = df_raw[pj2h_cols].copy()
    pj2h_df.columns = ['Week', 'Current_Active', 'Clients_With_Baselines', 'Percent']
    pj2h_df['Program'] = 'PJ2H'
    print("✓ PJ2H data extracted")
else:
    print("⚠ Could not find all PJ2H columns")
    pj2h_df = pd.DataFrame()

# Stack into tidy format
dataframes_to_concat = []
if len(port_df) > 0:
    dataframes_to_concat.append(port_df)
if len(pj2h_df) > 0:
    dataframes_to_concat.append(pj2h_df)

if len(dataframes_to_concat) == 0:
    raise ValueError("Could not extract any data from 'Notes of All the Numbers' sheet")

weekly_tidy = pd.concat(dataframes_to_concat, ignore_index=True)

# Clean data
weekly_tidy['WeekStart'] = weekly_tidy['Week'].apply(parse_week_start)
weekly_tidy = weekly_tidy.dropna(subset=['WeekStart'])

for col in ['Current_Active', 'Clients_With_Baselines', 'Percent']:
    weekly_tidy[col] = pd.to_numeric(weekly_tidy[col], errors='coerce')

weekly_tidy = weekly_tidy.dropna(subset=['Clients_With_Baselines'])
weekly_tidy = weekly_tidy.sort_values(['Program', 'WeekStart']).reset_index(drop=True)

# Compute weekly increments (how many MORE each week)
weekly_tidy['Weekly_Increment'] = weekly_tidy.groupby('Program')['Clients_With_Baselines'].diff()
weekly_tidy['Weekly_Increment'] = weekly_tidy['Weekly_Increment'].fillna(
    weekly_tidy['Clients_With_Baselines']
)

print("✓ Tidy weekly baseline table created")
print(f"  Shape: {weekly_tidy.shape}")
print(f"  Programs: {weekly_tidy['Program'].unique()}")
print(f"  Date range: {weekly_tidy['WeekStart'].min()} to {weekly_tidy['WeekStart'].max()}")
print(f"\nLast 8 rows:")
print(weekly_tidy.tail(8)[['Program', 'Week', 'Clients_With_Baselines',
                           'Weekly_Increment', 'Percent']].to_string(index=False))

# -------------------------------
# 2) Tidy 9.15 Progress Sheet
# -------------------------------
print("\n" + "=" * 80)
print("STEP 2: TIDYING 9.15 PROGRESS INTO PROGRAM-METRIC-VALUE TABLE")
print("=" * 80)

# Read with no header first to find structure
df_915_raw = pd.read_excel(FILEPATH, sheet_name='9.15 Progress', header=None)
print(f"Raw 9.15 Progress shape: {df_915_raw.shape}")
print("\nFirst 15 rows:")
print(df_915_raw.head(15).to_string())

# Find header row (contains PORT, PJ2H, etc.)
header_row = None
for i in range(min(15, len(df_915_raw))):
    row_str = ' '.join([str(x) for x in df_915_raw.iloc[i].values])
    if 'PORT' in row_str and ('PJ2H' in row_str or 'PS2H' in row_str):
        header_row = i
        print(f"\n✓ Found header row at index {i}")
        break

if header_row is None:
    header_row = 0
    print("⚠ Using first row as header")

# Re-read with correct header
df_915 = pd.read_excel(FILEPATH, sheet_name='9.15 Progress', header=header_row)
print(f"\nColumns after header fix: {list(df_915.columns)}")

# Build tidy program-metric-value table
outcomes_915 = []

# Identify metric column (usually first column with row labels)
metric_col = df_915.columns[0]

for idx, row in df_915.iterrows():
    metric_name = str(row[metric_col]).strip()

    # Skip if not a valid metric
    if metric_name in ['nan', '', 'None'] or metric_name.startswith('Unnamed'):
        continue

    # Extract values for each program
    for col in df_915.columns[1:]:  # Skip first column (metrics)
        col_name = str(col).upper().strip()

        # Determine program
        program = None
        if 'PORT' in col_name and 'REPORT' not in col_name:
            program = 'PORT'
        elif 'PJ2H' in col_name or 'PS2H' in col_name:
            program = 'PJ2H'
        elif 'BHOP' in col_name:
            program = 'BHOP'

        if program:
            try:
                value = pd.to_numeric(row[col], errors='coerce')
                if not pd.isna(value):
                    outcomes_915.append({
                        'Program': program,
                        'Metric': metric_name,
                        'Value': value,
                        'Sheet': '9.15 Progress'
                    })
            except:
                pass

outcomes_915_df = pd.DataFrame(outcomes_915)

print(f"\n✓ Tidy outcomes table created from 9.15 Progress")
print(f"  Shape: {outcomes_915_df.shape}")
print(f"  Programs: {outcomes_915_df['Program'].unique() if len(outcomes_915_df) > 0 else 'None'}")
print(f"  Unique metrics: {outcomes_915_df['Metric'].nunique() if len(outcomes_915_df) > 0 else 0}")

if len(outcomes_915_df) > 0:
    print("\nOutcomes by Program and Metric:")
    pivot = outcomes_915_df.pivot_table(
        index='Metric',
        columns='Program',
        values='Value',
        aggfunc='first'
    )
    print(pivot.to_string())

# -------------------------------
# 3) Extract 9.21 Progress (if exists)
# -------------------------------
print("\n" + "=" * 80)
print("STEP 3: CHECKING FOR ADDITIONAL PROGRESS SHEETS")
print("=" * 80)

outcomes_921_df = pd.DataFrame()

if '9.21 Progress' in xls.sheet_names:
    df_921_raw = pd.read_excel(FILEPATH, sheet_name='9.21 Progress', header=None)

    # Find header
    header_row_921 = None
    for i in range(min(15, len(df_921_raw))):
        row_str = ' '.join([str(x) for x in df_921_raw.iloc[i].values])
        if 'PORT' in row_str and ('PJ2H' in row_str or 'PS2H' in row_str):
            header_row_921 = i
            break

    if header_row_921 is None:
        header_row_921 = 0

    df_921 = pd.read_excel(FILEPATH, sheet_name='9.21 Progress', header=header_row_921)

    outcomes_921 = []
    metric_col = df_921.columns[0]

    for idx, row in df_921.iterrows():
        metric_name = str(row[metric_col]).strip()
        if metric_name in ['nan', '', 'None'] or metric_name.startswith('Unnamed'):
            continue

        for col in df_921.columns[1:]:
            col_name = str(col).upper().strip()
            program = None
            if 'PORT' in col_name and 'REPORT' not in col_name:
                program = 'PORT'
            elif 'PJ2H' in col_name or 'PS2H' in col_name:
                program = 'PJ2H'
            elif 'BHOP' in col_name:
                program = 'BHOP'

            if program:
                try:
                    value = pd.to_numeric(row[col], errors='coerce')
                    if not pd.isna(value):
                        outcomes_921.append({
                            'Program': program,
                            'Metric': metric_name,
                            'Value': value,
                            'Sheet': '9.21 Progress'
                        })
                except:
                    pass

    outcomes_921_df = pd.DataFrame(outcomes_921)
    print(f"✓ Found 9.21 Progress: {len(outcomes_921_df)} records")
else:
    print("⚠ No 9.21 Progress sheet found")

# Combine all outcomes
all_outcomes = []
if len(outcomes_915_df) > 0:
    outcomes_915_df['Date'] = pd.to_datetime('2025-09-15')
    all_outcomes.append(outcomes_915_df)
if len(outcomes_921_df) > 0:
    outcomes_921_df['Date'] = pd.to_datetime('2025-09-21')
    all_outcomes.append(outcomes_921_df)

if len(all_outcomes) > 0:
    outcomes_combined = pd.concat(all_outcomes, ignore_index=True)
    outcomes_combined = outcomes_combined.sort_values(['Program', 'Metric', 'Date'])
else:
    outcomes_combined = pd.DataFrame()

# -------------------------------
# 4) Visualize Weekly Trends
# -------------------------------
print("\n" + "=" * 80)
print("STEP 4: VISUALIZING WEEKLY BASELINE TRENDS")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top: Cumulative baselines
ax1 = axes[0]
programs = weekly_tidy['Program'].unique()
colors = {'PORT': '#10AC84', 'PJ2H': '#2E86DE', 'BHOP': '#EE5A6F'}

for prog in programs:
    data = weekly_tidy[weekly_tidy['Program'] == prog].copy()
    color = colors.get(prog, '#95A5A6')

    ax1.plot(data['WeekStart'], data['Clients_With_Baselines'],
             marker='o', linewidth=3, markersize=10, label=prog,
             color=color, alpha=0.8)

ax1.set_title('Cumulative Clients With Baselines Over Time',
              fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Week Starting', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Clients With Baselines', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Bottom: Weekly increments
ax2 = axes[1]

for prog in programs:
    data = weekly_tidy[weekly_tidy['Program'] == prog].copy()
    color = colors.get(prog, '#95A5A6')

    ax2.plot(data['WeekStart'], data['Weekly_Increment'],
             marker='o', linewidth=3, markersize=10, label=prog,
             color=color, alpha=0.8)

    # Add mean line
    mean_val = data['Weekly_Increment'].mean()
    ax2.axhline(y=mean_val, color=color, linestyle='--', alpha=0.5,
                linewidth=2, label=f'{prog} avg: {mean_val:.1f}/week')

ax2.set_title('Weekly New Baselines (Increments)',
              fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Week Starting', fontsize=13, fontweight='bold')
ax2.set_ylabel('New Baselines This Week', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'weekly_baseline_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: weekly_baseline_trends.png")
plt.close()

# -------------------------------
# 5) Project Baselines to November
# -------------------------------
print("\n" + "=" * 80)
print("STEP 5: PROJECTING BASELINES TO NOVEMBER")
print("=" * 80)

fig, ax = plt.subplots(figsize=(16, 9))

# Determine projection period (to end of November 2025)
last_date = weekly_tidy['WeekStart'].max()
target_date = pd.to_datetime('2025-11-30')
weeks_to_project = int((target_date - last_date).days / 7)

print(f"Last data point: {last_date.date()}")
print(f"Projecting {weeks_to_project} weeks to {target_date.date()}")

for prog in programs:
    data = weekly_tidy[weekly_tidy['Program'] == prog].copy()

    if len(data) < 3:
        continue

    # Create week index
    data = data.reset_index(drop=True)
    data['WeekIndex'] = range(len(data))

    X = data['WeekIndex'].values.reshape(-1, 1)
    y = data['Clients_With_Baselines'].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    slope = model.coef_[0]

    # Generate future indices
    future_idx = np.arange(len(data), len(data) + weeks_to_project)
    all_idx = np.concatenate([data['WeekIndex'].values, future_idx])
    all_pred = model.predict(all_idx.reshape(-1, 1))

    # Create date axis for plotting
    future_dates = [data['WeekStart'].iloc[-1] + timedelta(weeks=i + 1)
                    for i in range(weeks_to_project)]
    all_dates = list(data['WeekStart']) + future_dates

    color = colors.get(prog, '#95A5A6')

    # Plot actual data
    ax.scatter(data['WeekStart'], y, c=color, s=120, alpha=0.8,
               label=f'{prog} (actual)', zorder=3, edgecolors='white', linewidths=2)

    # Plot fitted line
    ax.plot(data['WeekStart'], model.predict(X), '-', c=color, linewidth=3, alpha=0.9)

    # Plot projection
    projection_dates = [data['WeekStart'].iloc[-1]] + future_dates
    projection_vals = [model.predict(X)[-1]] + list(all_pred[len(data):])
    ax.plot(projection_dates, projection_vals, '--', c=color, linewidth=3, alpha=0.7,
            label=f'{prog} projection: {slope:+.2f}/week (R²={r2:.3f})')

    # Confidence interval
    residuals = y - model.predict(X)
    std_err = np.std(residuals)
    ci = 1.96 * std_err

    ax.fill_between(projection_dates,
                    np.array(projection_vals) - ci,
                    np.array(projection_vals) + ci,
                    color=color, alpha=0.15)

    # Report
    current = y[-1]
    projected = all_pred[-1]
    print(f"\n{prog}:")
    print(f"  Current:          {current:.0f} clients")
    print(f"  Weekly growth:    {slope:+.2f} clients/week")
    print(f"  Nov 30 projection: {projected:.0f} clients")
    print(f"  Total growth:     {projected - current:+.0f} clients")
    print(f"  R² score:         {r2:.3f}")

# Add vertical line at projection start
ax.axvline(x=last_date, color='black', linestyle=':', linewidth=2.5,
           alpha=0.7, label='Projection starts', zorder=2)

ax.set_title('BASELINES: Projection to November 30, 2025',
             fontsize=17, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Clients With Baselines', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'baselines_projection_november.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: baselines_projection_november.png")
plt.close()

# -------------------------------
# 6) Project Outcomes to November
# -------------------------------
if len(outcomes_combined) > 0:
    print("\n" + "=" * 80)
    print("STEP 6: PROJECTING OUTCOMES TO NOVEMBER")
    print("=" * 80)

    # Filter for key outcome metrics
    outcome_keywords = ['Short Term Outcomes Completed', 'Long Term Outcomes Completed']
    outcomes_filtered = outcomes_combined[
        outcomes_combined['Metric'].str.contains('|'.join(outcome_keywords), case=False, na=False)
    ].copy()

    if len(outcomes_filtered) > 0:
        fig, ax = plt.subplots(figsize=(16, 9))

        metric_colors = {
            'Short Term': {'PORT': '#10AC84', 'PJ2H': '#2E86DE', 'BHOP': '#EE5A6F'},
            'Long Term': {'PORT': '#0A6F5C', 'PJ2H': '#1B4F91', 'BHOP': '#A63D4A'}
        }

        for prog in outcomes_filtered['Program'].unique():
            for outcome_type in ['Short Term', 'Long Term']:
                mask = (outcomes_filtered['Program'] == prog) & \
                       (outcomes_filtered['Metric'].str.contains(outcome_type, case=False))
                data = outcomes_filtered[mask].sort_values('Date')

                if len(data) < 2:
                    continue

                # Create time index
                data = data.reset_index(drop=True)
                data['TimeIndex'] = range(len(data))

                X = data['TimeIndex'].values.reshape(-1, 1)
                y = data['Value'].values

                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                r2 = r2_score(y, model.predict(X))
                slope = model.coef_[0]

                # Project 6 more periods
                future_idx = np.arange(len(data), len(data) + 6)
                all_idx = np.concatenate([data['TimeIndex'].values, future_idx])
                all_pred = model.predict(all_idx.reshape(-1, 1))

                color = metric_colors[outcome_type].get(prog, '#95A5A6')
                marker = 'o' if outcome_type == 'Short Term' else 's'

                # Plot
                ax.scatter(data['TimeIndex'], y, c=color, s=150, alpha=0.9,
                           label=f'{prog} {outcome_type} (actual)', marker=marker,
                           zorder=4, edgecolors='white', linewidths=2)

                ax.plot(data['TimeIndex'], model.predict(X), '-', c=color, linewidth=3, alpha=0.9)

                proj_idx = np.concatenate([[data['TimeIndex'].iloc[-1]], future_idx])
                proj_vals = np.concatenate([[model.predict(X)[-1]], all_pred[len(data):]])
                ax.plot(proj_idx, proj_vals, '--', c=color, linewidth=3, alpha=0.7,
                        label=f'{prog} {outcome_type} proj: {slope:+.1f}/period (R²={r2:.2f})')

                # CI
                residuals = y - model.predict(X)
                ci = 1.96 * np.std(residuals)
                ax.fill_between(proj_idx, proj_vals - ci, proj_vals + ci,
                                color=color, alpha=0.15)

                print(f"\n{prog} - {outcome_type}:")
                print(f"  Current:      {y[-1]:.0f}")
                print(f"  Growth rate:  {slope:+.2f}/period")
                print(f"  Nov projection: {all_pred[-1]:.0f}")
                print(f"  R²:           {r2:.3f}")

        ax.axvline(x=len(data) - 0.5, color='black', linestyle=':',
                   linewidth=2.5, alpha=0.7, label='Projection starts', zorder=2)

        ax.set_title('OUTCOMES: Projection to November 2025',
                     fontsize=17, fontweight='bold', pad=15)
        ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax.set_ylabel('Outcomes Completed', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + 'outcomes_projection_november.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: outcomes_projection_november.png")
        plt.close()
    else:
        print("⚠ No outcome metrics found matching criteria")
else:
    print("\n⚠ No outcomes data available for projection")

# -------------------------------
# 7) Final Summary
# -------------------------------
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\n BASELINES (Tidy Weekly Table):")
for prog in programs:
    prog_data = weekly_tidy[weekly_tidy['Program'] == prog]
    current = prog_data['Clients_With_Baselines'].iloc[-1]
    avg_increment = prog_data['Weekly_Increment'].mean()
    latest_pct = prog_data['Percent'].iloc[-1]
    print(f"  {prog}: {current:.0f} total | {avg_increment:.1f} avg new/week | {latest_pct:.1f}% complete")

if len(outcomes_combined) > 0:
    print("\n OUTCOMES (Tidy Program-Metric-Value Table):")
    summary = outcomes_combined.groupby(['Program', 'Metric'])['Value'].agg(['count', 'last'])
    for idx, row in summary.iterrows():
        prog, metric = idx
        print(f"  {prog} - {metric}: {row['last']:.0f} (from {row['count']} snapshots)")

print("\n ANALYSIS COMPLETE!")
print("=" * 80)
print("\n Files saved:")
print("  • weekly_baseline_trends.png")
print("  • baselines_projection_november.png")
if len(outcomes_combined) > 0:
    print("  • outcomes_projection_november.png")
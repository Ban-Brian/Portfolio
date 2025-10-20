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
SHEET_NAME = 'Dashboard Progress'  # Updated sheet name

print("=" * 80)
print("EJSL DASHBOARD ANALYSIS")
print("=" * 80)

# -------------------------------
# 1) Load Data from Dashboard Progress
# -------------------------------
print("\n📥 Loading data from Dashboard Progress sheet...")

try:
    # First, let's see what sheets are available
    xls = pd.ExcelFile(FILEPATH)
    print(f"Available sheets: {xls.sheet_names}")

    # Try to find the right sheet
    if SHEET_NAME not in xls.sheet_names:
        print(f"\n⚠️  '{SHEET_NAME}' not found!")
        print("Looking for similar sheet names...")
        for sheet in xls.sheet_names:
            if 'dashboard' in sheet.lower() or 'progress' in sheet.lower():
                SHEET_NAME = sheet
                print(f"✓ Using sheet: '{SHEET_NAME}'")
                break

    # Read the sheet to see structure
    df_raw = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=None)
    print(f"\n✓ Raw data loaded from '{SHEET_NAME}': {df_raw.shape}")
    print("\nFirst 15 rows:")
    print(df_raw.head(15).to_string())

    # Find the header row that contains "Date", "Program", etc.
    header_row = None
    for i in range(min(20, len(df_raw))):
        row_values = [str(v).lower().strip() for v in df_raw.iloc[i].values if pd.notna(v)]
        # Look for key column indicators
        if any(keyword in ' '.join(row_values) for keyword in ['date', 'program', 'port', 'pj2h', 'baseline']):
            # Check if this row has multiple non-null values (likely a header)
            non_null = df_raw.iloc[i].notna().sum()
            if non_null >= 3:
                header_row = i
                print(f"\n✓ Found potential header row at index {i}")
                print(f"  Row contents: {df_raw.iloc[i].tolist()}")
                break

    if header_row is None:
        print("\n⚠️  Could not automatically find header row")
        print("Showing first 20 rows for manual inspection:")
        for i in range(min(20, len(df_raw))):
            print(f"  Row {i}: {df_raw.iloc[i].tolist()}")

        # Ask user or make best guess
        header_row = 0
        print(f"\nUsing row {header_row} as header")

    # Re-read with correct header
    df = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=header_row)
    print(f"\n✓ Data loaded with headers: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Remove completely empty rows
    df = df.dropna(how='all')
    print(f"After removing empty rows: {df.shape}")

    if len(df) == 0:
        print("\n❌ ERROR: DataFrame is empty after loading!")
        print("This might mean:")
        print("  1. The header row is incorrect")
        print("  2. All data rows are being filtered out")
        print("  3. The sheet is empty or formatted unexpectedly")
        print("\nPlease check your Excel file structure.")
        raise ValueError("Empty dataframe - cannot proceed")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback

    traceback.print_exc()
    raise

# -------------------------------
# 2) Clean and Prepare Data
# -------------------------------
print("\n🧹 Cleaning data...")

# Clean column names
df.columns = [str(col).strip() for col in df.columns]

# Remove any completely empty rows
df = df.dropna(how='all')

print("\nFirst 5 data rows:")
print(df.head().to_string())

# Find the actual column names
print("\n🔍 Identifying columns...")
date_col = None
program_col = None
baselines_col = None
short_term_col = None
long_term_col = None

for col in df.columns:
    col_lower = col.lower()
    if 'date' in col_lower or col in ['8/26', '9/11', '9/15']:
        date_col = col
        print(f"  Date column: '{col}'")
    elif 'program' in col_lower:
        program_col = col
        print(f"  Program column: '{col}'")
    elif 'baseline' in col_lower and 'all' in col_lower:
        baselines_col = col
        print(f"  Baselines column: '{col}'")
    elif 'short' in col_lower and 'term' in col_lower:
        short_term_col = col
        print(f"  Short-term column: '{col}'")
    elif 'long' in col_lower and 'term' in col_lower:
        long_term_col = col
        print(f"  Long-term column: '{col}'")

# If columns not found, let user know what we have
if not all([date_col, program_col, baselines_col]):
    print("\n⚠️  Could not find all required columns!")
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
        print(f"      Sample values: {df[col].head(3).tolist()}")

    # Try to proceed with assumptions if we have any columns at all
    if len(df.columns) > 0:
        if date_col is None:
            date_col = df.columns[0]
            print(f"\n  Using column 0 as Date: '{date_col}'")
        if program_col is None and len(df.columns) > 1:
            program_col = df.columns[1]
            print(f"  Using column 1 as Program: '{program_col}'")
        if baselines_col is None:
            # Look for a column with "Baseline" in it
            for col in df.columns:
                if 'baseline' in col.lower():
                    baselines_col = col
                    print(f"  Using '{col}' as Baselines")
                    break
            # If still not found, try to find a numeric column
            if baselines_col is None:
                for col in df.columns[2:]:  # Skip first two (likely date and program)
                    if pd.api.types.is_numeric_dtype(df[col]):
                        baselines_col = col
                        print(f"  Using numeric column '{col}' as Baselines")
                        break
    else:
        raise ValueError("No columns found in dataframe!")

# Create standardized columns
df['Date'] = pd.to_datetime(df[date_col], format='%m/%d', errors='coerce')
# Infer year based on month sequence (if August-October, use 2025)
current_year = 2025
df['Date'] = df['Date'].apply(lambda x: x.replace(year=current_year) if pd.notna(x) else x)

df['Program'] = df[program_col].astype(str).str.strip()

# Convert numeric columns
if baselines_col:
    df['Baselines'] = pd.to_numeric(df[baselines_col], errors='coerce')
if short_term_col:
    df['ShortTerm'] = pd.to_numeric(df[short_term_col], errors='coerce')
if long_term_col:
    df['LongTerm'] = pd.to_numeric(df[long_term_col], errors='coerce')

# Remove rows with no date or program
df = df.dropna(subset=['Date', 'Program'])

# Sort
df = df.sort_values(['Program', 'Date']).reset_index(drop=True)

print(f"\n✓ Data cleaned: {len(df)} rows")
print(f"  Programs: {df['Program'].unique()}")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Calculate increments
df['Baseline_Increment'] = df.groupby('Program')['Baselines'].diff()
df['Baseline_Increment'] = df['Baseline_Increment'].fillna(df['Baselines'])

if 'ShortTerm' in df.columns:
    df['ShortTerm_Increment'] = df.groupby('Program')['ShortTerm'].diff()
if 'LongTerm' in df.columns:
    df['LongTerm_Increment'] = df.groupby('Program')['LongTerm'].diff()

print("\n📊 Sample of prepared data:")
print(df[['Date', 'Program', 'Baselines', 'Baseline_Increment']].head(10).to_string())

# -------------------------------
# 3) Visualize Baselines
# -------------------------------
print("\n" + "=" * 80)
print("📊 CREATING BASELINE VISUALIZATIONS")
print("=" * 80)

colors = {'PORT': '#10AC84', 'PJ2H': '#2E86DE', 'PS2H': '#2E86DE', 'BHOP': '#EE5A6F'}
programs = df['Program'].unique()

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top: Cumulative baselines
ax1 = axes[0]
for prog in programs:
    data = df[df['Program'] == prog].copy()
    color = colors.get(prog, '#95A5A6')

    ax1.plot(data['Date'], data['Baselines'],
             marker='o', linewidth=3, markersize=10, label=prog,
             color=color, alpha=0.8)

ax1.set_title('Cumulative Clients With Baselines Over Time',
              fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Clients With Baselines', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Bottom: Weekly increments
ax2 = axes[1]
for prog in programs:
    data = df[df['Program'] == prog].copy()
    color = colors.get(prog, '#95A5A6')

    ax2.plot(data['Date'], data['Baseline_Increment'],
             marker='o', linewidth=3, markersize=10, label=prog,
             color=color, alpha=0.8)

    mean_val = data['Baseline_Increment'].mean()
    ax2.axhline(y=mean_val, color=color, linestyle='--', alpha=0.5,
                linewidth=2, label=f'{prog} avg: {mean_val:.1f}/period')

ax2.set_title('New Baselines Each Period (How Many More)',
              fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
ax2.set_ylabel('New Baselines This Period', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'baselines_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: baselines_trends.png")
plt.close()

# -------------------------------
# 4) Project Baselines to November
# -------------------------------
print("\n" + "=" * 80)
print("📈 PROJECTING BASELINES TO NOVEMBER 30")
print("=" * 80)

fig, ax = plt.subplots(figsize=(16, 9))

last_date = df['Date'].max()
target_date = pd.to_datetime('2025-11-30')
periods_to_project = 8

print(f"Last data: {last_date.date()}")
print(f"Projecting to: {target_date.date()}\n")

for prog in programs:
    data = df[df['Program'] == prog].copy()

    if len(data) < 2:
        print(f"⚠️  {prog}: Not enough data for projection")
        continue

    data = data.reset_index(drop=True)
    data['TimeIndex'] = range(len(data))

    X = data['TimeIndex'].values.reshape(-1, 1)
    y = data['Baselines'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    slope = model.coef_[0]

    future_idx = np.arange(len(data), len(data) + periods_to_project)
    all_pred = model.predict(np.concatenate([X.flatten(), future_idx]).reshape(-1, 1))

    days_between = (data['Date'].iloc[-1] - data['Date'].iloc[-2]).days if len(data) > 1 else 7
    future_dates = [data['Date'].iloc[-1] + timedelta(days=days_between * (i + 1))
                    for i in range(periods_to_project)]

    color = colors.get(prog, '#95A5A6')

    # Actual data
    ax.scatter(data['Date'], y, c=color, s=150, alpha=0.8,
               label=f'{prog} (actual)', zorder=3, edgecolors='white', linewidths=2)

    # Fitted line
    ax.plot(data['Date'], model.predict(X), '-', c=color, linewidth=3, alpha=0.9)

    # Projection
    projection_dates = [data['Date'].iloc[-1]] + future_dates
    projection_vals = [model.predict(X)[-1]] + list(all_pred[len(data):])
    ax.plot(projection_dates, projection_vals, '--', c=color, linewidth=3, alpha=0.7,
            label=f'{prog} projection: {slope:+.2f}/period (R²={r2:.3f})')

    # Confidence interval
    residuals = y - model.predict(X)
    std_err = np.std(residuals)
    ci = 1.96 * std_err

    ax.fill_between(projection_dates,
                    np.array(projection_vals) - ci,
                    np.array(projection_vals) + ci,
                    color=color, alpha=0.15)

    current = y[-1]
    projected = all_pred[-1]
    print(f"{prog}:")
    print(f"  Current:           {current:.0f} clients")
    print(f"  Growth rate:       {slope:+.2f} clients/period")
    print(f"  Nov 30 projection: {projected:.0f} clients")
    print(f"  Expected growth:   {projected - current:+.0f} clients")
    print(f"  R² (fit quality):  {r2:.3f}\n")

ax.axvline(x=last_date, color='black', linestyle=':', linewidth=2.5,
           alpha=0.7, label='Projection starts', zorder=2)

ax.set_title('BASELINES: Trend Progression & November 30 Projection',
             fontsize=17, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Clients With Baselines', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'baselines_projection_november.png', dpi=300, bbox_inches='tight')
print("✓ Saved: baselines_projection_november.png")
plt.close()

# -------------------------------
# 5) Short-Term Outcomes
# -------------------------------
if 'ShortTerm' in df.columns:
    print("\n" + "=" * 80)
    print("📊 SHORT-TERM OUTCOMES ANALYSIS")
    print("=" * 80)

    df_short = df[df['ShortTerm'].notna()].copy()

    if len(df_short) > 0:
        fig, ax = plt.subplots(figsize=(16, 9))

        for prog in df_short['Program'].unique():
            data = df_short[df_short['Program'] == prog].copy()

            if len(data) < 2:
                continue

            data = data.reset_index(drop=True)
            data['TimeIndex'] = range(len(data))

            X = data['TimeIndex'].values.reshape(-1, 1)
            y = data['ShortTerm'].values

            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            slope = model.coef_[0]

            future_idx = np.arange(len(data), len(data) + periods_to_project)
            all_pred = model.predict(np.concatenate([X.flatten(), future_idx]).reshape(-1, 1))

            color = colors.get(prog, '#95A5A6')

            ax.scatter(data['TimeIndex'], y, c=color, s=150, alpha=0.9,
                       label=f'{prog} (actual)', marker='o', zorder=4,
                       edgecolors='white', linewidths=2)

            ax.plot(data['TimeIndex'], model.predict(X), '-',
                    c=color, linewidth=3, alpha=0.9)

            proj_idx = np.concatenate([[data['TimeIndex'].iloc[-1]], future_idx])
            proj_vals = np.concatenate([[model.predict(X)[-1]], all_pred[len(data):]])
            ax.plot(proj_idx, proj_vals, '--', c=color, linewidth=3, alpha=0.7,
                    label=f'{prog} projection: {slope:+.1f}/period (R²={r2:.3f})')

            residuals = y - model.predict(X)
            ci = 1.96 * np.std(residuals)
            ax.fill_between(proj_idx, proj_vals - ci, proj_vals + ci,
                            color=color, alpha=0.15)

            print(f"{prog}:")
            print(f"  Current:       {y[-1]:.0f}")
            print(f"  Growth rate:   {slope:+.2f}/period")
            print(f"  Projection:    {all_pred[-1]:.0f}")
            print(f"  R²:            {r2:.3f}\n")

        ax.axvline(x=len(data) - 0.5, color='black', linestyle=':',
                   linewidth=2.5, alpha=0.7, label='Projection starts', zorder=2)

        ax.set_title('SHORT-TERM OUTCOMES: Trend Progression & November Projection',
                     fontsize=17, fontweight='bold', pad=15)
        ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax.set_ylabel('Short-Term Outcomes Completed', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + 'shortterm_outcomes_projection.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: shortterm_outcomes_projection.png")
        plt.close()

# -------------------------------
# 6) Long-Term Outcomes
# -------------------------------
if 'LongTerm' in df.columns:
    print("\n" + "=" * 80)
    print("📊 LONG-TERM OUTCOMES ANALYSIS")
    print("=" * 80)

    df_long = df[df['LongTerm'].notna()].copy()

    if len(df_long) > 0:
        fig, ax = plt.subplots(figsize=(16, 9))

        long_colors = {'PORT': '#0A6F5C', 'PJ2H': '#1B4F91', 'PS2H': '#1B4F91', 'BHOP': '#A63D4A'}

        for prog in df_long['Program'].unique():
            data = df_long[df_long['Program'] == prog].copy()

            if len(data) < 2:
                continue

            data = data.reset_index(drop=True)
            data['TimeIndex'] = range(len(data))

            X = data['TimeIndex'].values.reshape(-1, 1)
            y = data['LongTerm'].values

            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            slope = model.coef_[0]

            future_idx = np.arange(len(data), len(data) + periods_to_project)
            all_pred = model.predict(np.concatenate([X.flatten(), future_idx]).reshape(-1, 1))

            color = long_colors.get(prog, '#95A5A6')

            ax.scatter(data['TimeIndex'], y, c=color, s=150, alpha=0.9,
                       label=f'{prog} (actual)', marker='s', zorder=4,
                       edgecolors='white', linewidths=2)

            ax.plot(data['TimeIndex'], model.predict(X), '-',
                    c=color, linewidth=3, alpha=0.9)

            proj_idx = np.concatenate([[data['TimeIndex'].iloc[-1]], future_idx])
            proj_vals = np.concatenate([[model.predict(X)[-1]], all_pred[len(data):]])
            ax.plot(proj_idx, proj_vals, '--', c=color, linewidth=3, alpha=0.7,
                    label=f'{prog} projection: {slope:+.1f}/period (R²={r2:.3f})')

            residuals = y - model.predict(X)
            ci = 1.96 * np.std(residuals)
            ax.fill_between(proj_idx, proj_vals - ci, proj_vals + ci,
                            color=color, alpha=0.15)

            print(f"{prog}:")
            print(f"  Current:       {y[-1]:.0f}")
            print(f"  Growth rate:   {slope:+.2f}/period")
            print(f"  Projection:    {all_pred[-1]:.0f}")
            print(f"  R²:            {r2:.3f}\n")

        ax.axvline(x=len(data) - 0.5, color='black', linestyle=':',
                   linewidth=2.5, alpha=0.7, label='Projection starts', zorder=2)

        ax.set_title('LONG-TERM OUTCOMES: Trend Progression & November Projection',
                     fontsize=17, fontweight='bold', pad=15)
        ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax.set_ylabel('Long-Term Outcomes Completed', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + 'longterm_outcomes_projection.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: longterm_outcomes_projection.png")
        plt.close()

# -------------------------------
# 7) Summary Report
# -------------------------------
print("\n" + "=" * 80)
print("📋 FINAL SUMMARY")
print("=" * 80)

print("\n🎯 BASELINES:")
for prog in programs:
    prog_data = df[df['Program'] == prog]
    if len(prog_data) > 0:
        current = prog_data['Baselines'].iloc[-1]
        avg_increment = prog_data['Baseline_Increment'].mean()
        print(f"  {prog}: {current:.0f} total | +{avg_increment:.1f} avg/period")

if 'ShortTerm' in df.columns:
    print("\n📈 SHORT-TERM OUTCOMES:")
    df_short = df[df['ShortTerm'].notna()]
    for prog in df_short['Program'].unique():
        prog_data = df_short[df_short['Program'] == prog]
        if len(prog_data) > 0:
            current = prog_data['ShortTerm'].iloc[-1]
            print(f"  {prog}: {current:.0f} total")

if 'LongTerm' in df.columns:
    print("\n📈 LONG-TERM OUTCOMES:")
    df_long = df[df['LongTerm'].notna()]
    for prog in df_long['Program'].unique():
        prog_data = df_long[df_long['Program'] == prog]
        if len(prog_data) > 0:
            current = prog_data['LongTerm'].iloc[-1]
            print(f"  {prog}: {current:.0f} total")

print("\n📁 FILES SAVED:")
print("  • baselines_trends.png")
print("  • baselines_projection_november.png")
if 'ShortTerm' in df.columns and len(df[df['ShortTerm'].notna()]) > 0:
    print("  • shortterm_outcomes_projection.png")
if 'LongTerm' in df.columns and len(df[df['LongTerm'].notna()]) > 0:
    print("  • longterm_outcomes_projection.png")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
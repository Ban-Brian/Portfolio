import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

FILEPATH = '/Users/brianbutler/Desktop/EJSL/Dashboard EJS Draft Pers.xlsx'
OUTPUT_DIR = '/Users/brianbutler/Desktop/EJSL/'
SHEET_NAME = 'Dashboard Progress'

print("=" * 70)
print("EJSL DASHBOARD COMPREHENSIVE FORECAST ANALYSIS")
print("=" * 70)

print("\nLoading Excel file...")
xls = pd.ExcelFile(FILEPATH)
print(f"Available sheets: {xls.sheet_names}")

df_raw = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=None)
print(f"\nRaw data shape: {df_raw.shape}")

print("\nSearching for header row...")
header_row = None
for i in range(min(30, len(df_raw))):
    row_text = ' '.join([str(v).lower() for v in df_raw.iloc[i].values if pd.notna(v)])
    if any(keyword in row_text for keyword in ['date', 'program', 'baseline', 'port', 'pj2h']):
        non_null_count = df_raw.iloc[i].notna().sum()
        if non_null_count >= 2:
            header_row = i
            print(f"Found header at row {i}: {df_raw.iloc[i].tolist()}")
            break

if header_row is None:
    print("Could not find header. Using row 0.")
    header_row = 0

df = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=header_row)
df = df.dropna(how='all')

print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns")
print(f"Columns found: {list(df.columns)}")

date_cols = []
program_cols = []
numeric_cols = []

for col in df.columns:
    col_lower = str(col).lower()
    sample_values = df[col].dropna().head(5)

    if 'date' in col_lower or '/' in str(col):
        date_cols.append(col)
    elif 'program' in col_lower or any(str(v) in ['PORT', 'PJ2H', 'PS2H', 'BHOP'] for v in sample_values):
        program_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df[col]) or all(pd.to_numeric(sample_values, errors='coerce').notna()):
        numeric_cols.append(col)

print(f"\nIdentified columns:")
print(f"  Date columns: {date_cols}")
print(f"  Program columns: {program_cols}")
print(f"  Numeric columns: {numeric_cols}")

# Use the first of each type found
date_col = date_cols[0] if date_cols else df.columns[0]
program_col = program_cols[0] if program_cols else df.columns[1] if len(df.columns) > 1 else None
data_cols = numeric_cols if numeric_cols else df.columns[2:].tolist()

print(f"\nUsing:")
print(f"  Date: {date_col}")
print(f"  Program: {program_col}")
print(f"  Data columns: {data_cols}")

# Clean and prepare data
print("\nCleaning data...")
df_clean = df.copy()

# Parse dates
df_clean['Date'] = pd.to_datetime(df_clean[date_col], format='%m/%d', errors='coerce')
df_clean['Date'] = df_clean['Date'].apply(lambda x: x.replace(year=2025) if pd.notna(x) else x)

# Get programs
if program_col:
    df_clean['Program'] = df_clean[program_col].astype(str).str.strip()
else:
    df_clean['Program'] = 'Unknown'

# Convert numeric columns
for col in data_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Remove empty rows
df_clean = df_clean.dropna(subset=['Date'])
df_clean = df_clean.sort_values(['Program', 'Date']).reset_index(drop=True)

print(f"Cleaned data: {len(df_clean)} rows")
print(f"Programs: {df_clean['Program'].unique().tolist()}")
print(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")

print("\nData preview:")
print(df_clean.head(10).to_string())

# Program colors
colors = {
    'PORT': '#10AC84',
    'PJ2H': '#2E86DE',
    'PS2H': '#2E86DE',
    'BHOP': '#EE5A6F',
    'Unknown': '#95A5A6'
}

# Forecast settings
target_date = pd.to_datetime('2025-11-30')
periods_to_project = 8

# Create forecasts for each numeric column
for data_col in data_cols:
    # Skip if column is all NaN
    if df_clean[data_col].isna().all():
        print(f"\nSkipping {data_col} - no data")
        continue

    print(f"\n{'=' * 70}")
    print(f"FORECASTING: {data_col}")
    print(f"{'=' * 70}")

    # Filter data with values for this column
    df_analysis = df_clean[df_clean[data_col].notna()].copy()

    if len(df_analysis) == 0:
        print(f"No data for {data_col}")
        continue

    fig, ax = plt.subplots(figsize=(14, 8))

    for program in df_analysis['Program'].unique():
        prog_data = df_analysis[df_analysis['Program'] == program].copy()

        if len(prog_data) < 2:
            print(f"\n{program}: Need at least 2 data points")
            continue

        # Prepare regression data
        prog_data['TimeIndex'] = range(len(prog_data))
        X = prog_data['TimeIndex'].values.reshape(-1, 1)
        y = prog_data[data_col].values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        # Generate predictions
        future_indices = np.arange(len(prog_data), len(prog_data) + periods_to_project)
        all_indices = np.concatenate([X.flatten(), future_indices])
        predictions = model.predict(all_indices.reshape(-1, 1))

        # Generate future dates
        if len(prog_data) > 1:
            days_between = (prog_data['Date'].iloc[-1] - prog_data['Date'].iloc[-2]).days
        else:
            days_between = 7

        future_dates = [prog_data['Date'].iloc[-1] + timedelta(days=days_between * (i + 1))
                        for i in range(periods_to_project)]

        color = colors.get(program, '#95A5A6')

        # Plot actual data
        ax.scatter(prog_data['Date'], y, c=color, s=100, alpha=0.8,
                   label=f'{program} (actual)', zorder=3, edgecolors='white', linewidths=1.5)

        # Plot trend line
        ax.plot(prog_data['Date'], model.predict(X), '-', c=color,
                linewidth=2.5, alpha=0.9)

        # Plot projection
        projection_dates = [prog_data['Date'].iloc[-1]] + future_dates
        projection_values = [predictions[len(prog_data) - 1]] + list(predictions[len(prog_data):])
        ax.plot(projection_dates, projection_values, '--', c=color,
                linewidth=2.5, alpha=0.7, label=f'{program} forecast')

        # Print summary
        current = y[-1]
        projected = predictions[-1]
        growth_rate = model.coef_[0]

        print(f"\n{program}:")
        print(f"  Current value:        {current:.1f}")
        print(f"  Growth rate:          {growth_rate:+.2f} per period")
        print(f"  Nov 30 projection:    {projected:.1f}")
        print(f"  Expected change:      {projected - current:+.1f}")
        print(f"  Model fit (R²):       {r2:.3f}")

    # Mark projection start
    last_date = df_analysis['Date'].max()
    ax.axvline(x=last_date, color='black', linestyle=':', linewidth=2,
               alpha=0.6, label='Projection starts')

    # Format plot
    ax.set_title(f'{data_col} - Forecast to November 30, 2025',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel(data_col, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    safe_filename = data_col.replace(' ', '_').replace('/', '_')
    output_path = f"{OUTPUT_DIR}forecast_{safe_filename}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
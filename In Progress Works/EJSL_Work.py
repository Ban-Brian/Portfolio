import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Configuration
FILEPATH = '/Users/brianbutler/Desktop/EJSL/Dashboard EJS Draft Pers.xlsx'
OUTPUT_DIR = '/Users/brianbutler/Desktop/EJSL/'
SHEET_NAME = 'Dashboard Progress'

print("EJSL Dashboard Forecast Analysis")
print("=" * 60)

# Load data
print("\nLoading data...")
df_raw = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=None)

# Find header row
header_row = 0
for i in range(min(20, len(df_raw))):
    row_text = ' '.join([str(v).lower() for v in df_raw.iloc[i].values if pd.notna(v)])
    if 'date' in row_text or 'program' in row_text:
        header_row = i
        break

# Load with correct header
df = pd.read_excel(FILEPATH, sheet_name=SHEET_NAME, header=header_row)
df = df.dropna(how='all')

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Identify columns
date_col = [col for col in df.columns if 'date' in str(col).lower()][0]
program_col = [col for col in df.columns if 'program' in str(col).lower()][0]
baseline_col = [col for col in df.columns if 'baseline' in str(col).lower()][0]

# Clean data
df['Date'] = pd.to_datetime(df[date_col], format='%m/%d', errors='coerce')
df['Date'] = df['Date'].apply(lambda x: x.replace(year=2025) if pd.notna(x) else x)
df['Program'] = df[program_col].astype(str).str.strip()
df['Baselines'] = pd.to_numeric(df[baseline_col], errors='coerce')

df = df.dropna(subset=['Date', 'Program', 'Baselines'])
df = df.sort_values(['Program', 'Date']).reset_index(drop=True)

print(f"\nCleaned data: {len(df)} rows")
print(f"Programs: {df['Program'].unique().tolist()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Program colors
colors = {
    'PORT': '#10AC84',
    'PJ2H': '#2E86DE',
    'PS2H': '#2E86DE',
    'BHOP': '#EE5A6F'
}

# Create forecast
target_date = pd.to_datetime('2025-11-30')
periods_to_project = 8

fig, ax = plt.subplots(figsize=(14, 8))

print("\nForecasting to November 30, 2025:")
print("-" * 60)

for program in df['Program'].unique():
    program_data = df[df['Program'] == program].copy()

    if len(program_data) < 2:
        continue

    # Prepare data for regression
    program_data['TimeIndex'] = range(len(program_data))
    X = program_data['TimeIndex'].values.reshape(-1, 1)
    y = program_data['Baselines'].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate projections
    future_indices = np.arange(len(program_data), len(program_data) + periods_to_project)
    all_indices = np.concatenate([X.flatten(), future_indices])
    predictions = model.predict(all_indices.reshape(-1, 1))

    # Generate future dates
    days_between = (program_data['Date'].iloc[-1] - program_data['Date'].iloc[-2]).days
    future_dates = [program_data['Date'].iloc[-1] + timedelta(days=days_between * (i + 1))
                    for i in range(periods_to_project)]

    color = colors.get(program, '#95A5A6')

    # Plot actual data
    ax.scatter(program_data['Date'], y, c=color, s=100, alpha=0.8,
               label=f'{program} (actual)', zorder=3)

    # Plot trend line
    ax.plot(program_data['Date'], model.predict(X), '-', c=color,
            linewidth=2.5, alpha=0.9)

    # Plot projection
    projection_dates = [program_data['Date'].iloc[-1]] + future_dates
    projection_values = [predictions[len(program_data) - 1]] + list(predictions[len(program_data):])
    ax.plot(projection_dates, projection_values, '--', c=color,
            linewidth=2.5, alpha=0.7, label=f'{program} projection')

    # Print forecast summary
    current = y[-1]
    projected = predictions[-1]
    growth_rate = model.coef_[0]

    print(f"\n{program}:")
    print(f"  Current baselines:    {current:.0f}")
    print(f"  Growth rate:          {growth_rate:+.2f} per period")
    print(f"  Nov 30 projection:    {projected:.0f}")
    print(f"  Expected growth:      {projected - current:+.0f}")

# Mark projection start
last_date = df['Date'].max()
ax.axvline(x=last_date, color='black', linestyle=':', linewidth=2,
           alpha=0.6, label='Projection starts')

# Format plot
ax.set_title('Baselines Forecast to November 30, 2025',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Clients with Baselines', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'forecast_results.png', dpi=300, bbox_inches='tight')
print(f"\n\nSaved: {OUTPUT_DIR}forecast_results.png")
print("\nAnalysis complete!")
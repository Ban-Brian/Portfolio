import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Tester 4
# -------------------------------
# 1) Locate Excel file
# -------------------------------
# Get the folder where this script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build full path to the Excel file
file_path = os.path.join(script_dir, "Dashboard EJS Draft.xlsx")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Cannot find Excel file at {file_path}")

# Load the Excel workbook
xl = pd.ExcelFile(file_path, engine="openpyxl")
# Show all sheets in the Excel file
print("Available sheets:", xl.sheet_names)

# -------------------------------
# 2) Parse weekly % table
# -------------------------------
# Read the sheet that contains weekly percentage data
weekly_df = xl.parse("Total Client % with All", header=0)
# Remove extra spaces from column names
weekly_df.columns = weekly_df.columns.str.strip()

# List of columns we need
keep = ["Week (2025)", "PJ2H %", "PORT %", "BHOP %"]

# Make sure the expected columns exist
for col in keep:
    if col not in weekly_df.columns:
        raise KeyError(f"Expected column '{col}' not found in weekly data")

# Remove any rows that are missing data in the key columns
weekly_df = weekly_df.dropna(subset=keep)
# Ensure the week column is treated as text
weekly_df["Week (2025)"] = weekly_df["Week (2025)"].astype(str)
# Ensure the percentage columns are treated as numbers
weekly_df[keep[1:]] = weekly_df[keep[1:]].astype(float)

# Calculate the week-to-week change for each column
weekly_df["PJ2H Δ"] = weekly_df["PJ2H %"].diff()
weekly_df["PORT Δ"] = weekly_df["PORT %"].diff()
weekly_df["BHOP Δ"] = weekly_df["BHOP %"].diff()

# Print these changes so we can see trends
print("\nWeek-over-week changes:")
print(weekly_df[["Week (2025)", "PJ2H Δ", "PORT Δ", "BHOP Δ"]])

# -------------------------------
# 3) Projection function
# -------------------------------
# Create an array of week indices for plotting
weeks = np.arange(len(weekly_df))
# Define future weeks to project beyond current data
future_weeks = np.arange(len(weekly_df), len(weekly_df)+10)

# Function to create a linear trend projection
def project_trend(series):
    lr = LinearRegression()                 # Create a linear regression model
    lr.fit(weeks.reshape(-1, 1), series.values)  # Fit model to existing data
    idx = np.concatenate([weeks, future_weeks]) # Combine current + future weeks
    return idx, lr.predict(idx.reshape(-1, 1))  # Return projected values

# Compute projections for each category
pj2h_idx, pj2h_proj = project_trend(weekly_df["PJ2H %"])
port_idx, port_proj = project_trend(weekly_df["PORT %"])
bhop_idx, bhop_proj = project_trend(weekly_df["BHOP %"])

# -------------------------------
# 4) Load outcomes from "Progress" sheets
# -------------------------------
# Find sheets containing 'Progress' in their name
prog_sheets = [s for s in xl.sheet_names if "Progress" in s]
if len(prog_sheets) < 2:
    raise ValueError("Need at least two sheets containing 'Progress' in the name")

# Pick the last two sheets for comparison
sheet_prev, sheet_latest = prog_sheets[-2], prog_sheets[-1]
print(f"Using '{sheet_prev}' and '{sheet_latest}' for outcomes comparison")

# Function to safely extract short and long outcomes
def load_outcomes(sheet_name):
    # Read first 20 rows to find numbers
    df = xl.parse(sheet_name, nrows=20)

    # Convert all cells to numbers; text or errors become NaN
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # Flatten the table to a 1D array and remove empty cells
    numeric_values = df_numeric.values.flatten()
    numeric_values = numeric_values[~np.isnan(numeric_values)]

    # Make sure we found at least 2 numbers
    if len(numeric_values) < 2:
        raise ValueError(f"Sheet '{sheet_name}' does not contain at least 2 numeric outcome values")

    # Take the first two numbers as short and long outcomes
    short, long = numeric_values[0], numeric_values[1]
    return short, long

# Extract previous and latest outcomes
short_prev, long_prev = load_outcomes(sheet_prev)
short_latest, long_latest = load_outcomes(sheet_latest)

# -------------------------------
# 5) Plot baseline % projections with outcomes
# -------------------------------
plt.figure(figsize=(12, 6))

# Plot projected trends as dashed lines
plt.plot(pj2h_idx, pj2h_proj, "--", c="blue",  label="PJ2H % proj")
plt.plot(port_idx, port_proj, "--", c="green", label="PORT % proj")
plt.plot(bhop_idx, bhop_proj, "--", c="red",   label="BHOP % proj")

# Plot actual weekly data as scatter points
plt.scatter(weeks, weekly_df["PJ2H %"], c="blue")
plt.scatter(weeks, weekly_df["PORT %"],  c="green")
plt.scatter(weeks, weekly_df["BHOP %"],  c="red")

# Add outcome markers at the end of the baseline series
x0, x1 = len(weeks)-2, len(weeks)-1
plt.plot([x0, x1], [short_prev,  short_latest],  "o-", c="purple", label="Short-term outcomes")
plt.plot([x0, x1], [ long_prev,   long_latest],  "o-", c="orange", label="Long-term outcomes")

plt.title("Baseline % Trends + Outcome Completions Projection")
plt.xlabel("Week index")
plt.ylabel("Value (%) or Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_projection.png")  # Save figure to file
plt.close()

# -------------------------------
# 6) Plot outcomes projection separately
# -------------------------------
out_idx = np.arange(2)              # indices for prev and latest
future_out = np.arange(2, 12)       # indices for 10 future points
all_out_idx = np.concatenate([out_idx, future_out])

# Fit linear regression for short-term outcomes
lr_short = LinearRegression().fit(out_idx.reshape(-1,1), [short_prev, short_latest])
# Fit linear regression for long-term outcomes
lr_long  = LinearRegression().fit(out_idx.reshape(-1,1), [long_prev,  long_latest])

# Predict future outcomes
short_proj = lr_short.predict(all_out_idx.reshape(-1,1))
long_proj  = lr_long.predict(all_out_idx.reshape(-1,1))

# Plot projections
plt.figure(figsize=(8, 5))
plt.plot(all_out_idx, short_proj, "--", c="purple", label="Short-term proj")
plt.scatter(out_idx, [short_prev, short_latest], c="purple")
plt.plot(all_out_idx, long_proj, "--", c="orange", label="Long-term proj")
plt.scatter(out_idx, [ long_prev,  long_latest], c="orange")

# Label the x-axis
xticks = ["prev", "latest"] + [f"+{i}" for i in range(1,11)]
plt.xticks(all_out_idx, xticks, rotation=45)
plt.title("Outcome Completions Trend + Projection")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outcomes_trend_projection.png")  # Save figure
plt.close()

print("\nScript completed successfully! Plots saved as 'combined_projection.png' and 'outcomes_trend_projection.png'.")
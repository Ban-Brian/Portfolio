import os
import pandas as pd
from openpyxl import load_workbook

# --- File path ---
file_path = "/Users/brianbutler/Desktop/EJSL/Dashboard EJS Draft.xlsx"

# --- Check if file exists ---
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Excel file not found at: {file_path}")

# --- Load the workbook ---
wb = load_workbook(filename=file_path, data_only=True)  # data_only=True reads evaluated values

# --- Load "Notes of All the Numbers" sheet into DataFrame ---
sheet_name = "Notes of All the Numbers"
df = pd.DataFrame(wb[sheet_name].values)

# --- Extract PORT weekly data ---
port_weekly = []
for i, row in df.iloc[2:21].iterrows():  # rows 2-20
    week_label = row[0]
    if week_label and "to" in str(week_label):
        port_weekly.append({
            "weekNum": i - 1,
            "week": week_label,
            "activeAdmissions": row[1] if pd.notnull(row[1]) else 0,
            "clientsWithBaselines": row[2] if pd.notnull(row[2]) else 0,
            "percent": row[3] if pd.notnull(row[3]) else 0
        })

# --- Extract PJ2H weekly data ---
pj2h_weekly = []
for i, row in df.iloc[2:21].iterrows():  # rows 2-20
    week_label = row[5]
    if week_label and "to" in str(week_label):
        pj2h_weekly.append({
            "weekNum": i - 1,
            "week": week_label,
            "activeAdmissions": row[6] if pd.notnull(row[6]) else 0,
            "clientsWithBaselines": row[7] if pd.notnull(row[7]) else 0,
            "percent": row[8] if pd.notnull(row[8]) else 0
        })

# --- Function to print weekly progression ---
def print_weekly_progress(data, program_name):
    print(f"\n=== {program_name} PROGRAM - WEEKLY PROGRESSION ANALYSIS ===\n")
    print("Baseline Completion Rates:")
    for idx, week in enumerate(data):
        change = ""
        if idx > 0:
            diff = week["clientsWithBaselines"] - data[idx-1]["clientsWithBaselines"]
            change = f"(+{diff})" if diff >= 0 else f"({diff})"
        print(f"Week {week['weekNum']}: {week['clientsWithBaselines']} baselines {change} - {week['percent']*100:.1f}% completion rate")

# --- Print PORT and PJ2H weekly progression ---
print_weekly_progress(port_weekly, "PORT")
print_weekly_progress(pj2h_weekly, "PJ2H")

# --- Calculate trend analysis ---
port_avg_growth = (port_weekly[-1]["clientsWithBaselines"] - port_weekly[0]["clientsWithBaselines"]) / (len(port_weekly) - 1)
pj2h_avg_growth = (pj2h_weekly[-1]["clientsWithBaselines"] - pj2h_weekly[0]["clientsWithBaselines"]) / (len(pj2h_weekly) - 1)

print("\n=== TREND ANALYSIS ===")
print(f"PORT average weekly change: {port_avg_growth:.2f} baselines/week")
print(f"PJ2H average weekly change: {pj2h_avg_growth:.2f} baselines/week")

# --- Extract "Total Values of Programs" sheet ---
total_sheet_name = "Total Values of Programs"
df_total = pd.DataFrame(wb[total_sheet_name].values)
df_total.columns = df_total.iloc[0]  # first row as header
df_total = df_total[1:]  # drop header row

# --- Group by Program and summarize ---
print("\n=== TOTAL VALUES SUMMARY ===")
grouped = df_total.groupby("Program")
for program, group in grouped:
    print(f"\n{program}:")
    for _, row in group.iterrows():
        metric = row["Metric"]
        value = row["Value"]
        print(f"  {metric}: {value}")

import os
import re
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# === FILE CONFIG ===
folder_path = "/Users/brianbutler/Desktop/EJSL"
output_csv = os.path.join(folder_path, "peer_study_weekly_data.csv")
baseline_png = os.path.join(folder_path, "baseline_trends.png")
outcomes_png = os.path.join(folder_path, "outcome_trends.png")

# === HELPER FUNCTIONS ===
def extract_numbers_from_text(text):
    """Extract all numbers (ints) from a text block in order."""
    text = text.replace("", "-").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return [int(num) for num in re.findall(r"\d+", text)]

def extract_percents_from_text(text):
    """Extract all percentages from a text block in order."""
    text = text.replace("", "-").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return [int(num) for num in re.findall(r"(\d+)%", text)]

def parse_pdf_weekly(text):
    """Extract a single row of numeric data per date range from text."""
    entries = []

    # Extract date range (start date)
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s*[-–]?\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    if not date_match:
        return []

    start_date = datetime.strptime(date_match.group(1), "%m/%d/%Y")

    # Extract numbers
    nums = extract_numbers_from_text(text)
    percents = extract_percents_from_text(text)

    # Map numbers to columns (adjust indices based on your PDF)
    active_admissions = nums[0] if len(nums) > 0 else 0
    active_baselines = nums[1] if len(nums) > 1 else 0
    all_baselines = nums[2] if len(nums) > 2 else 0
    short_term = nums[3] if len(nums) > 3 else 0
    long_term = nums[4] if len(nums) > 4 else 0

    percent_active = percents[0] if len(percents) > 0 else 0
    percent_all = percents[1] if len(percents) > 1 else 0

    entries.append({
        "week_start_date": start_date,
        "active_admissions": active_admissions,
        "active_baselines": active_baselines,
        "all_baselines": all_baselines,
        "percent_active": percent_active,
        "percent_all": percent_all,
        "short_term": short_term,
        "long_term": long_term
    })
    return entries

# === READ ALL PDFs ===
def read_all_pdfs_weekly(folder):
    all_data = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder, filename)
            print(f"📘 Reading {pdf_path} ...")
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    # Extract all numeric data per page
                    all_data.extend(parse_pdf_weekly(text))
    return pd.DataFrame(all_data)

# === DATA PROCESSING ===
def compute_growth(df, col):
    df = df.sort_values("week_start_date").reset_index(drop=True)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[f"{col}_change"] = df[col].diff().fillna(0)
    return df

def project_trend(df, col, weeks_forward=8):
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    x = np.arange(len(df))
    y = df[col].fillna(0)
    if len(df) < 2 or y.sum() == 0:
        return y, []
    coeffs = np.polyfit(x, y, 1)
    proj = np.poly1d(coeffs)
    proj_x = np.arange(len(df) + weeks_forward)
    proj_y = proj(proj_x)
    future_dates = [df["week_start_date"].iloc[-1] + timedelta(weeks=i) for i in range(1, weeks_forward + 1)]
    return proj_y, future_dates

# === MAIN SCRIPT ===
df_all = read_all_pdfs_weekly(folder_path)

if df_all.empty:
    raise ValueError("⚠️ No valid data found in PDFs.")

# Ensure numeric columns
numeric_cols = [
    "active_admissions", "active_baselines", "all_baselines",
    "percent_active", "percent_all", "short_term", "long_term"
]
for col in numeric_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)

# Save CSV
df_all.to_csv(output_csv, index=False)
print(f"✅ Weekly data saved to {output_csv}")
print(df_all.head())

# === VISUALIZATION ===

# --- Baselines trend chart ---
df_baselines = compute_growth(df_all, "active_baselines")
plt.figure(figsize=(11, 6))
plt.plot(df_baselines["week_start_date"], df_baselines["active_baselines"], "o-", label="Actual")
proj_y, future_dates = project_trend(df_baselines, "active_baselines")
if future_dates:
    future_proj_y = proj_y[-len(future_dates):]
    plt.plot(future_dates, future_proj_y, "--", label="Projected")
plt.title("Active Baselines Progression Week by Week")
plt.xlabel("Week Start Date")
plt.ylabel("Active Clients With Baselines")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(baseline_png)
plt.close()

# --- Short-term vs Long-term outcomes ---
plt.figure(figsize=(11, 6))
plt.plot(df_all["week_start_date"], df_all["short_term"], "o-", label="Short-Term")
plt.plot(df_all["week_start_date"], df_all["long_term"], "o--", label="Long-Term", alpha=0.6)
plt.title("Short-Term vs Long-Term Outcomes Week by Week")
plt.xlabel("Week Start Date")
plt.ylabel("Outcomes Completed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outcomes_png)
plt.close()

print(f"✅ Baseline trend chart saved to {baseline_png}")
print(f"✅ Outcome comparison chart saved to {outcomes_png}")

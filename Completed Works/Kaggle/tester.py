import pandas as pd
import numpy as np

sub = pd.read_csv("/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/submission.csv")

print("QUICK CHECK:")
print(f"Shape: {sub.shape}")
print(f"\nFirst 30 predictions:")
print(sub.head(30))
print(f"\nStats:")
print(sub['accident_risk'].describe())
print(f"\nUnique values: {sub['accident_risk'].nunique()}")
print(f"Min: {sub['accident_risk'].min()}")
print(f"Max: {sub['accident_risk'].max()}")
print(f"Mean: {sub['accident_risk'].mean()}")

# Check if predictions look reasonable
print(f"\nDo predictions vary? {sub['accident_risk'].std() > 0.01}")
print(f"Standard deviation: {sub['accident_risk'].std()}")

# Sample random predictions
print(f"\n20 random predictions:")
print(sub.sample(20)['accident_risk'].values)
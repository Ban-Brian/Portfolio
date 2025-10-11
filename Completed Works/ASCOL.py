import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate

 
# Summary Statistics
 
population = 51269
avg_household_size = 5.0
num_households = round(population / avg_household_size)
cost_of_living_index = 116.35
median_income = 41742

 
# DataFrames

spending_data = pd.DataFrame({
    'Category': ['Food', 'Housing', 'Apparel', 'Alcoholic Beverages', 'Other'],
    'Aggregate (USD)': [82134000, 87883000, 15084000, 3811000, 281747000],
    'Per Household (USD)': [8010, 8573, 1471, 372, 27496]
})

commodity_prices = pd.DataFrame({
    'Commodity': ['Apple (per lb)', 'Beef Chuck (per lb)', 'Coors Light (12 fl. oz.)', 'Bottle Gas (30 lbs)'],
    'Price (USD)': [1.88, 4.94, 1.66, 25.00]
})

index_data = pd.DataFrame({
    'Category': ['Food', 'Housing', 'Transportation'],
    'Index Value': [140.8, 125.5, 122.8]
})

 
# Combined Visualization Dashboard
 
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

# Plot 1: Per-Household Spending
sns.barplot(
    y='Category', x='Per Household (USD)',
    data=spending_data, color='steelblue', edgecolor='black',
    ax=axes[0]
)
axes[0].set_title('Per-Household Spending by Category (2022)', fontsize=12)
axes[0].set_xlabel('USD')
axes[0].set_ylabel('')

# Plot 2: Commodity Prices
sns.barplot(
    y='Commodity', x='Price (USD)',
    data=commodity_prices, color='seagreen', edgecolor='black',
    ax=axes[1]
)
axes[1].set_title('Commodity Prices in American Samoa (2022)', fontsize=12)
axes[1].set_xlabel('USD')
axes[1].set_ylabel('')

# Plot 3: Cost of Living Index
sns.barplot(
    x='Category', y='Index Value',
    data=index_data, palette='Oranges', edgecolor='black',
    ax=axes[2]
)
axes[2].set_title('Cost of Living Index by Category (Base 2016)', fontsize=12)
axes[2].set_ylabel('Index (Base = 100)')
axes[2].set_xlabel('')

plt.suptitle('Cost of Living Dashboard - American Samoa (2022)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cost_of_living_dashboard.png")
plt.show()

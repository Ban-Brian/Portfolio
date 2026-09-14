import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Example data structure
data = {
    'Revenue': {'values': [100, 120, 140], 'type': 'currency'},
    'Expenses': {'values': [80, 90, 110], 'type': 'currency'},
    'Profit Margin': {'values': [0.10, 0.15, 0.20], 'type': 'percentage'}
}

# Simulated user input (replace with actual input mechanism)
user_input = {
    'Revenue': '200, 220, 240',
    'Expenses': 'bad, 100, 120',  # Invalid input
    'Profit Margin': '0.12, 0.18, 0.22'
}

errors = []
for label, raw_input in user_input.items():
    try:
        values = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
        if not values:
            raise ValueError("No numeric values entered.")
        data[label]['values'] = values
    except ValueError as e:
        errors.append(f"Invalid input for '{label}': {e}")

if errors:
    for err in errors:
        print(err)

# Plot updated data
plt.figure(figsize=(8, 5))
x = list(range(1, len(next(iter(data.values()))['values']) + 1))
for label, info in data.items():
    y = info['values']
    plt.plot(x, y, marker='o', label=label)

plt.xlabel('Time Step')
plt.title('Financial and Percentage Variables Over Time')
types = set(info['type'] for info in data.values())
ax = plt.gca()
if len(types) == 1:
    var_type = types.pop()
    if var_type == 'currency':
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        plt.ylabel('Value (USD)')
    elif var_type == 'percentage':
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.ylabel('Value (%)')
    else:
        plt.ylabel('Value')
else:
    plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
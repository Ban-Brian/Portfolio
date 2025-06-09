import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import matplotlib.ticker as mticker

# Initial data with metadata indicating if the variable is a percentage or a raw number (currency)
data = {
    'Revenue': {'values': [100, 120, 140], 'type': 'currency'},
    'Expenses': {'values': [80, 90, 110], 'type': 'currency'},
    'Profit Margin': {'values': [0.10, 0.15, 0.20], 'type': 'percentage'}
}

def submit():
    """
    Retrieve user input from the entry fields, validate, and update the data dictionary.
    Handles errors by showing a message box and skipping invalid inputs.
    After updating, re-plot the data.
    """
    for label, entry in entries.items():
        raw_input = entry.get()
        try:
            # Split input by comma, strip spaces, and convert to float
            values = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
            if not values:
                raise ValueError("No numeric values entered.")
            data[label]['values'] = values
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid input for '{label}': {e}\nPlease enter comma-separated numeric values.")
            return  # Stop processing on first error
    print("Updated data:", {k: v['values'] for k, v in data.items()})
    plot_data()

def plot_data():
    """
    Plot the data with appropriate y-axis formatting depending on variable type.
    Currency values are formatted with dollar signs; percentages as percentages.
    """
    plt.figure(figsize=(8, 5))
    # Determine the number of time steps from the first variable's values
    x = list(range(1, len(next(iter(data.values()))['values']) + 1))

    for label, info in data.items():
        y = info['values']
        plt.plot(x, y, marker='o', label=label)

    plt.xlabel('Time Step')
    plt.title('Financial and Percentage Variables Over Time')

    # Create a combined y-axis formatter that can handle multiple lines with different formats
    # Since matplotlib doesn't support multiple y-axis formats on the same axis,
    # we will format the y-tick labels as currency if all are currency or as percentage if all are percentage,
    # else default formatting.
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
        # Mixed types: just label as Value and no special formatting
        plt.ylabel('Value')

    plt.legend()
    plt.tight_layout()
    plt.show()

root = tk.Tk()
root.title("Financial and Percentage Variable Input")

entries = {}

# Create input fields for each variable with a dropdown to select variable type
for i, (var_name, info) in enumerate(data.items()):
    tk.Label(root, text=var_name).grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(root, width=30)
    # Pre-fill with current data values as comma-separated string
    entry.insert(0, ', '.join(str(v) for v in info['values']))
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[var_name] = entry

    # Dropdown to select variable type (percentage or currency)
    var_type_var = tk.StringVar(value=info['type'])
    def update_type(var=var_name, var_type_var=var_type_var):
        data[var]['type'] = var_type_var.get()
    option_menu = tk.OptionMenu(root, var_type_var, 'currency', 'percentage', command=lambda _: update_type())
    option_menu.grid(row=i, column=2, padx=5, pady=5)

tk.Button(root, text="Submit", command=submit).grid(row=len(data), column=0, columnspan=3, pady=10)

root.mainloop()
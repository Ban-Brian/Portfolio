import sys
import statistics
import json

class DataProcessor:
    def __init__(self, data):
        """
        Initialize the processor with data.
        :param data: A list of numbers.
        """
        self.data = data

    def calculate_sum(self):
        """Calculate the sum of the data."""
        total = sum(self.data)
        print(f"Debug: Calculated sum = {total}")
        return total

    def calculate_average(self):
        """Calculate the average of the data."""
        if len(self.data) > 0:
            avg = self.calculate_sum() / len(self.data)
            print(f"Debug: Calculated average = {avg}")
            return avg
        print("Debug: Data list is empty. Cannot calculate average.")
        return None

    def calculate_median(self):
        """Calculate the median of the data."""
        if len(self.data) > 0:
            med = statistics.median(self.data)
            print(f"Debug: Calculated median = {med}")
            return med
        print("Debug: Data list is empty. Cannot calculate median.")
        return None

    def sort_data(self, reverse=False):
        """
        Sort the data.
        :param reverse: If True, sort in descending order; otherwise ascending.
        """
        sorted_list = sorted(self.data, reverse=reverse)
        order = "descending" if reverse else "ascending"
        print(f"Debug: Data sorted in {order} order = {sorted_list}")
        return sorted_list

    def to_json(self):
        """
        Export all statistics and the data itself as a JSON string.
        """
        stats = {
            "data": self.data, 
            "sum": self.calculate_sum(),
            "average": self.calculate_average(),
            "median": self.calculate_median()
        }
        json_output = json.dumps(stats, indent=4)
        print(f"Debug: JSON output generated = {json_output}")
        return json_output

def get_data_from_user():
    """
    Prompt the user for a series of numbers separated by spaces.
    Convert them to floats and return the list.
    """
    while True:
        user_input = input("Enter a series of numbers separated by spaces: ").strip()
        if not user_input:
            print("No input detected. Please try again.")
            continue
        try:
            # Convert each input to a float
            data = [float(num) for num in user_input.split()]
            print(f"Debug: User input converted to list = {data}")
            return data
        except ValueError as e:
            print("Error: Please ensure all entries are numbers. Try again.")
            print(f"Debug: Exception details: {e}")

def main():
    print("Welcome to the Data Processor Application!")
    data = get_data_from_user()

    processor = DataProcessor(data)

    while True:
        print("\nSelect an operation to perform:")
        print("1. Calculate Sum")
        print("2. Calculate Average")
        print("3. Calculate Median")
        print("4. Sort Data")
        print("5. Export All Statistics as JSON")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            result = processor.calculate_sum()
            print("The sum of the numbers is:", result)
        elif choice == "2":
            result = processor.calculate_average()
            if result is not None:
                print("The average of the numbers is:", result)
            else:
                print("Unable to calculate average due to insufficient data.")
        elif choice == "3":
            result = processor.calculate_median()
            if result is not None:
                print("The median of the numbers is:", result)
            else:
                print("Unable to calculate median due to insufficient data.")
        elif choice == "4":
            order = input("Enter 'asc' for ascending or 'desc' for descending order: ").strip().lower()
            reverse_order = order == 'desc'
            sorted_data = processor.sort_data(reverse=reverse_order)
            print("The sorted data is:", sorted_data)
        elif choice == "5":
            json_stats = processor.to_json()
            print("Statistics in JSON format:")
            print(json_stats)
        elif choice == "6":
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option from the menu.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
        sys.exit(0)

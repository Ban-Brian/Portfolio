import os

class ToDoListApp:
    def __init__(self, filename="todo_list.txt"):
        self.tasks = []
        self.filename = filename
        self.load_tasks()

    def load_tasks(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                self.tasks = [line.strip() for line in file]
        else:
            print("No previous task file found. Starting fresh!")

    def save_tasks(self):
        with open(self.filename, 'w') as file:
            for task in self.tasks:
                file.write(task + '\n')

    def add_task(self, task):
        self.tasks.append(task)
        print(f"Task '{task}' added successfully!")

    def remove_task(self, index):
        if index <= 0 or index > len(self.tasks):
            print("Invalid task number. Please choose a valid task to remove.")
            return
        removed_task = self.tasks.pop(index - 1)
        print(f"Task '{removed_task}' removed successfully!")

    def view_tasks(self):
        if not self.tasks:
            print("Your to-do list is empty!")
        else:
            print("\nYour To-Do List:")
            for i, task in enumerate(self.tasks, start=1):
                print(f"{i}. {task}")
        print()

    def main_menu(self):
        while True:
            print("=== To-Do List Menu ===")
            print("1. View Tasks")
            print("2. Add Task")
            print("3. Remove Task")
            print("4. Save and Exit")

            try:
                choice = int(input("Enter your choice (1-4): "))
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 4.\n")
                continue

            if choice == 1:
                self.view_tasks()
            elif choice == 2:
                task = input("Enter the task to add: ")
                self.add_task(task)
            elif choice == 3:
                self.view_tasks()
                try:
                    index = int(input("Enter the task number to remove: "))
                    self.remove_task(index)
                except ValueError:
                    print("Invalid input. Please enter a valid number.\n")
            elif choice == 4:
                self.save_tasks()
                print("Tasks saved. Goodbye!")
                break
            else:
                print("Invalid choice. Please choose a number between 1 and 4.\n")


if __name__ == "__main__":
    app = ToDoListApp()
    app.main_menu()

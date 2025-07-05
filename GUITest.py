import pandas as pd
import tkinter as tk
from tkinter import messagebox

class MissingValueApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x800")
        self.root.title("Student Performance Missing Value Checker")
        self.df = None

        # Fixed file path
        self.file_path = "Student_performance_data _.csv"  # Change to your actual CSV filename

        # GUI Elements
        self.label = tk.Label(root, text="Missing Value Report", font=("Arial", 14))
        self.label.pack(pady=10)

        self.output_text = tk.Text(root, height=25, width=100)
        self.output_text.pack(pady=10)

        # Call the checking function automatically
        self.check_missing()

    def check_missing(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            return
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Loaded file: {self.file_path}\n")

        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()

        self.output_text.insert(tk.END, "\nMissing values per column:\n")
        self.output_text.insert(tk.END, f"{missing_counts}\n")

        if total_missing > 0:
            self.output_text.insert(tk.END, f"\nTotal missing values: {total_missing}\n")
            self.output_text.insert(tk.END, "Cleaning data...\n")
            self.df = self.df.dropna()
            self.output_text.insert(tk.END, f"New shape after cleaning: {self.df.shape}\n")

            # Save cleaned data
            self.df.to_csv("Cleaned_StudentsPerformance.csv", index=False)
            self.output_text.insert(tk.END, "Cleaned data saved as 'Cleaned_StudentsPerformance.csv'\n")
        else:
            self.output_text.insert(tk.END, "No missing values found.\n")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MissingValueApp(root)
    root.mainloop()

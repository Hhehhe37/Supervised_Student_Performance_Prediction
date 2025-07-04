import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

class MissingValueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Missing Value Checker")
        self.df = None

        # GUI Elements
        self.label = tk.Label(root, text="Choose a CSV file to check for missing values", font=("Arial", 12))
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load CSV File", command=self.load_file, width=30)
        self.load_button.pack(pady=5)

        self.check_button = tk.Button(root, text="Check & Clean Missing Values", command=self.check_missing, state=tk.DISABLED, width=30)
        self.check_button.pack(pady=5)

        self.output_text = tk.Text(root, height=15, width=70)
        self.output_text.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Loaded file: {file_path}\n")
                self.check_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def check_missing(self):
        if self.df is None:
            return
        
        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()

        self.output_text.insert(tk.END, "\nMissing values per column:\n")
        self.output_text.insert(tk.END, f"{missing_counts}\n")

        if total_missing > 0:
            self.output_text.insert(tk.END, f"\nTotal missing values: {total_missing}\n")
            self.output_text.insert(tk.END, "Cleaning data...\n")
            self.df = self.df.dropna()
            self.output_text.insert(tk.END, f"New shape: {self.df.shape}\n")

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

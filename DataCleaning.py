import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class DataCleaningWindow:
    def __init__(self, window): 
        self.root = window
        self.root.title("🧹 Student Performance Missing Value Checker")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f5f7fa")

        self.df = None
        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")

        # Main frame with padding and white bg for content area
        main_frame = ttk.Frame(self.root, padding=20, style="Card.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title label
        title_label = ttk.Label(main_frame, text="Missing Value Report", font=("Segoe UI", 18, "bold"))
        title_label.pack(pady=(0, 15))

        # Treeview Frame (to hold treeview + scrollbar nicely)
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview widget
        self.tree = ttk.Treeview(tree_frame, columns=("Column", "Missing Count"), show="headings", height=15)
        self.tree.heading("Column", text="Column")
        self.tree.heading("Missing Count", text="Missing Values")
        self.tree.column("Column", width=500, anchor=tk.W)
        self.tree.column("Missing Count", width=150, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar for treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Status label
        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 12), foreground="#228B22")
        self.status_label.pack(pady=15)

        # Load and process data automatically
        self.load_and_process_data()

    # -- Your existing functions unchanged --
    def load_and_process_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            return
        
        self.remove_incorrect_data()
        self.check_missing()

    def assign_grade_class(self, gpa):
        if gpa >= 3.5:
            return 0  # A
        elif gpa >= 3.0:
            return 1  # B
        elif gpa >= 2.5:
            return 2  # C
        elif gpa >= 2.0:
            return 3  # D
        else:
            return 4  # F

    def remove_incorrect_data(self):
        df = self.df.copy()
        df['ExpectedGrade'] = df['GPA'].apply(self.assign_grade_class)
        correct_df = df[df['GradeClass'] == df['ExpectedGrade']].copy()
        self.df = correct_df.drop(columns=['ExpectedGrade'])
        output_path = os.path.join(OUTPUT_FOLDER, "Corrected_StudentsPerformance.csv")
        self.df.to_csv(output_path, index=False)
        self.status_label.config(text=f"Incorrect data removed. Records before: {len(df)}, after: {len(self.df)}.")

    def check_missing(self):
        self.tree.delete(*self.tree.get_children())  # Clear previous content

        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()

        for col, miss_count in missing_counts.items():
            self.tree.insert("", tk.END, values=(col, miss_count))

        if total_missing > 0:
            output_path = os.path.join(OUTPUT_FOLDER, "Cleaned_StudentsPerformance.csv")
            self.df = self.df.dropna()
            self.df.to_csv(output_path, index=False)
            self.status_label.config(text=f"Total missing values: {total_missing}. Cleaned data saved as '{output_path}'.")
        else:
            self.status_label.config(text="No missing values found.")


# Add ttk style for card-like frame appearance (optional)
def setup_styles():
    style = ttk.Style()
    style.theme_use('default')
    style.configure("Card.TFrame", background="#ffffff", relief="raised", borderwidth=1)
    style.configure("Treeview", font=("Segoe UI", 11))
    style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))

if __name__ == "__main__":
    setup_styles()
    root = tk.Tk()
    app = DataCleaningWindow(root)
    root.mainloop()

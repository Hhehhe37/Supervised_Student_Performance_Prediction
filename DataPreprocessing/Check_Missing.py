import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from DataPreprocessing.Check_Duplicate import CheckDuplicateWindow
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class CheckMissingWindow:
    def __init__(self, window, callback=None): 
        self.callback = callback
        self.root = window
        self.root.title("🧹 Student Performance Missing Value Checker")
        self.root.geometry("900x600")
        self.root.configure(bg="#f5f7fa")

        self.df = None
        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")
        self.missing_msg = ""
        self.duplicate_msg = ""

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

        next_button = ttk.Button(main_frame, text="Next ➡️ Check Duplicates", command=self.open_duplicate_window)
        next_button.pack(pady=10)

        # Load and process data automatically
        self.load_and_process_data()

    # -- Your existing functions unchanged --
    def load_and_process_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            return
        
        self.check_missing()

        if self.callback:
            self.callback(self.missing_msg)

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
            self.missing_msg = f"Missing values cleaned: {total_missing}"
        else:
            self.missing_msg = "No missing values found."

        self.status_label.config(text=self.missing_msg)

    def open_duplicate_window(self):
        self.root.destroy()
        new_window = tk.Tk()
        CheckDuplicateWindow(new_window)
        new_window.mainloop()
        

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
    app = CheckMissingWindow(root)
    root.mainloop()

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from DataPreprocessing.Check_Duplicate import CheckDuplicateWindow
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class CheckMissingWindow:
    def __init__(self, window, callback=None):   # Initialize the window with optional callback function
        self.callback = callback  # Store callback function for later use
        self.root = window  # Set the root window
        self.root.title("🧹 Student Performance Missing Value Checker")  # Set window title with emoji
        self.root.geometry("900x600")  # Set window dimensions
        self.root.configure(bg="#f5f7fa")  # Set background color

        self.df = None  # Initialize dataframe as None
        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")  # Set file path
        self.missing_msg = ""  # Initialize missing message
        self.duplicate_msg = ""  # Initialize duplicate message

        # Main frame with padding and white bg for content area
        main_frame = ttk.Frame(self.root, padding=20, style="Card.TFrame")  # Create main frame with padding
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)  # Pack frame with padding

        # Title label
        title_label = ttk.Label(main_frame, text="Missing Value Report", font=("Segoe UI", 18, "bold"))  # Create title label
        title_label.pack(pady=(0, 15))  # Pack title with vertical padding

        # Treeview Frame (to hold treeview + scrollbar nicely)
        tree_frame = ttk.Frame(main_frame)  # Create frame for treeview
        tree_frame.pack(fill=tk.BOTH, expand=True)  # Pack frame to expand

        # Treeview widget
        self.tree = ttk.Treeview(tree_frame, columns=("Column", "Missing Count"), show="headings", height=15)  # Create treeview
        self.tree.heading("Column", text="Column")  # Set column heading
        self.tree.heading("Missing Count", text="Missing Values")
        self.tree.column("Column", width=500, anchor=tk.W)  # Set column width and alignment
        self.tree.column("Missing Count", width=150, anchor=tk.CENTER)  # Set count column width and alignment
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Pack treeview to expand

        # Vertical scrollbar for treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)  # Create scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Pack scrollbar on right side
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Status label
        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 12), foreground="#228B22")  # Create status label
        self.status_label.pack(pady=15)  # Pack status label with vertical padding

        next_button = ttk.Button(main_frame, text="Next ➡️ Check Duplicates", command=self.open_duplicate_window)  # Create next button
        next_button.pack(pady=10)  # Pack button with vertical padding

        # Load and process data automatically
        self.load_and_process_data()  # Call method to load and process data

    # -- Your existing functions unchanged --
    def load_and_process_data(self):  # Method to load and process data
        try:
            self.df = pd.read_csv(self.file_path)  # Read CSV file into dataframe
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")  # Show error if file loading fails
            return  # Return if error occurs
        
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

import pandas as pd
import tkinter as tk
from tkinter import ttk
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class StatisticsWindow:
    def __init__(self, window):
        self.root = window
        self.root.title("📊 Student Performance Statistics")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f5f7fa")

        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")
        self.df = None

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        style = ttk.Style()
        style.configure("Treeview", font=("Segoe UI", 13))               # content font
        style.configure("Treeview.Heading", font=("Segoe UI", 16, "bold")) 
        title_label = ttk.Label(main_frame, text="Student Performance Statistics", font=("Segoe UI", 20, "bold"))
        title_label.pack(pady=(0, 15))

        # Status label
        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 20), foreground="#228B22")
        self.status_label.pack(pady=10)

        # Numeric Treeview
        numeric_label = ttk.Label(main_frame, text="Numeric Attributes", font=("Segoe UI", 20, "bold"))
        numeric_label.pack(pady=(15, 5))

        self.numeric_tree = ttk.Treeview(main_frame, columns=("Column","Count","Mean","Median","Std","Min","25%","50%","75%","Max"),
                                         show="headings", height=8)
        for col in self.numeric_tree["columns"]:
            self.numeric_tree.heading(col, text=col)
            self.numeric_tree.column(col, width=100, anchor=tk.CENTER)
        self.numeric_tree.pack(fill=tk.X, expand=True)

        # Categorical Treeview
        categorical_label = ttk.Label(main_frame, text="Categorical Attributes", font=("Segoe UI", 25, "bold"))
        categorical_label.pack(pady=(15, 5))

        self.cat_tree = ttk.Treeview(main_frame, columns=("Column","Count","Unique","Top","Freq"), show="headings", height=12)
        for col in self.cat_tree["columns"]:
            self.cat_tree.heading(col, text=col)
            self.cat_tree.column(col, width=150, anchor=tk.CENTER)
        self.cat_tree.pack(fill=tk.X, expand=True)

        # Load and display
        self.load_and_display_statistics()

    def load_and_display_statistics(self):   # ✅ now inside class
        try:
            self.df = pd.read_csv(self.file_path)
            self.df.columns = self.df.columns.str.strip()  # strip whitespace
        except Exception as e:
            self.status_label.config(text=f"❌ Failed to load file: {e}", foreground="red")
            return

        # --- Clear old rows before inserting new ---
        for item in self.numeric_tree.get_children():
            self.numeric_tree.delete(item)
        for item in self.cat_tree.get_children():
            self.cat_tree.delete(item)

        # --- Numeric statistics ---
        numeric_cols = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]  # safeguard
        if numeric_cols:
            numeric_df = self.df[numeric_cols]
            numeric_stats = numeric_df.describe().T
            numeric_stats["median"] = numeric_df.median()

            for col in numeric_df.columns:
                row = numeric_stats.loc[col]
                self.numeric_tree.insert("", tk.END, values=(
                    col,
                    f"{row['count']:.0f}",
                    f"{row['mean']:.2f}",
                    f"{row['median']:.2f}",
                    f"{row['std']:.2f}",
                    f"{row['min']:.2f}",
                    f"{row['25%']:.2f}",
                    f"{row['50%']:.2f}",
                    f"{row['75%']:.2f}",
                    f"{row['max']:.2f}",
                ))

        # --- Categorical statistics ---
        categorical_cols = [col for col in self.df.columns if col not in numeric_cols]
        if categorical_cols:
            self.df[categorical_cols] = self.df[categorical_cols].astype("category")

            for col in categorical_cols:
                col_data = self.df[col]
                mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else "N/A"
                top_freq = col_data.value_counts().iloc[0] if not col_data.value_counts().empty else "N/A"

                self.cat_tree.insert("", tk.END, values=(
                    col,
                    col_data.count(),   # count of non-null values
                    col_data.nunique(), # number of unique categories
                    mode_val,           # most frequent category
                    top_freq            # frequency of that category
                ))

        self.status_label.config(text="✅ Statistics loaded successfully!", foreground="#228B22")


if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticsWindow(root)
    root.mainloop()

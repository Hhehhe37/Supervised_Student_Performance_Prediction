import pandas as pd
import tkinter as tk
from tkinter import ttk
import os
from DataPreprocessing.Process_incorrect_data import RemoveIncorrectDataWindow

class CheckDuplicateWindow:
    def __init__(self, window): 
        self.root = window
        self.root.title("🔁 Duplicate Record Checker")
        self.root.geometry("600x400")
        self.root.configure(bg="#f5f7fa")

        self.file_path = os.path.join("CSV_Files", "Student_performance_data _.csv")

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="Duplicate Records", font=("Segoe UI", 18, "bold"))
        title_label.pack(pady=(0, 15))

        # Frame for table (will only be packed if duplicates exist)
        self.tree_frame = ttk.Frame(main_frame)

        self.tree = ttk.Treeview(self.tree_frame, show="headings")
        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Status message label
        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 12), foreground="#555")
        self.status_label.pack(pady=10)

        next_button = ttk.Button(main_frame, text="Next ➡️ Remove Incorrect Data", command=self.open_remove_incorrect)
        next_button.pack(pady=10)

        self.load_and_check_duplicates()

    def load_and_check_duplicates(self):
        try:
            df = pd.read_csv(self.file_path)
            duplicate_df = df[df.duplicated()]

            if not duplicate_df.empty:
                # Only show Treeview if there are duplicates
                self.tree_frame.pack(fill=tk.BOTH, expand=True)
                self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                self.tree["columns"] = list(duplicate_df.columns)
                for col in duplicate_df.columns:
                    self.tree.heading(col, text=col)
                    self.tree.column(col, width=100)

                for _, row in duplicate_df.iterrows():
                    self.tree.insert("", tk.END, values=list(row))

                self.status_label.config(text=f"{len(duplicate_df)} duplicate records found.")
            else:
                # Do NOT show treeview if no duplicates
                self.status_label.config(text="No duplicate records found.")
        except Exception as e:
            self.status_label.config(text=f"Error loading file:\n{e}")

    def open_remove_incorrect(self):
            self.root.destroy()
            new_root = tk.Tk()
            RemoveIncorrectDataWindow(new_root)
            new_root.mainloop()

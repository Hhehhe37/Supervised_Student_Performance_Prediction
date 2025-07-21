import pandas as pd
import tkinter as tk
from tkinter import ttk
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class RemoveIncorrectDataWindow:
    def __init__(self, window):
        self.root = window
        self.root.title("❌ Remove Incorrect GradeClass Data")
        self.root.geometry("900x600")
        self.root.configure(bg="#f5f7fa")

        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")
        self.df = None

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Incorrect Data Removal Report", font=("Segoe UI", 18, "bold"))
        title_label.pack(pady=(0, 15))

        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 12), foreground="#228B22")
        self.status_label.pack(pady=10)

        self.remove_incorrect_data()

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
        try:
            df = pd.read_csv(self.file_path)
            df['ExpectedGrade'] = df['GPA'].apply(self.assign_grade_class)
            before_count = len(df)
            correct_df = df[df['GradeClass'] == df['ExpectedGrade']].copy()
            correct_df = correct_df.drop(columns=['ExpectedGrade'])
            after_count = len(correct_df)

            output_path = os.path.join(OUTPUT_FOLDER, "Corrected_StudentsPerformance.csv")
            correct_df.to_csv(output_path, index=False)

            self.status_label.config(text=f"Incorrect records removed: {before_count - after_count}\nRecords after cleaning: {after_count}")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

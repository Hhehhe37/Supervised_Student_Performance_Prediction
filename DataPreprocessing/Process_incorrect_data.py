import pandas as pd
import tkinter as tk
from tkinter import ttk
import os

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class ReplaceIncorrectDataWindow:
    def __init__(self, window):
        self.root = window
        self.root.title("❌ Replace Incorrect GradeClass Data")
        self.root.geometry("900x600")
        self.root.configure(bg="#f5f7fa")

        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")
        self.df = None

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Incorrect Data Replace Report", font=("Segoe UI", 18, "bold"))
        title_label.pack(pady=(0, 15))

        self.status_label = ttk.Label(main_frame, text="", font=("Segoe UI", 12), foreground="#228B22")
        self.status_label.pack(pady=10)

        self.replace_incorrect_data()

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

    def replace_incorrect_data(self):
        try:
            df = pd.read_csv(self.file_path)
            df['ExpectedGrade'] = df['GPA'].apply(self.assign_grade_class)

            # Find records that are incorrect
            incorrect_mask = df['GradeClass'] != df['ExpectedGrade']
            incorrect_count = incorrect_mask.sum()

            # Count changes per grade
            changed_counts = df[incorrect_mask].groupby('ExpectedGrade').size().to_dict()
            grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            changed_counts_readable = {grade_mapping[k]: v for k, v in changed_counts.items()}

            # Replace incorrect GradeClass values
            df['GradeClass'] = df['ExpectedGrade']
            df = df.drop(columns=['ExpectedGrade'])

            # Save corrected data
            output_path = os.path.join(OUTPUT_FOLDER, "Corrected_StudentsPerformance.csv")
            df.to_csv(output_path, index=False)

            # Prepare a clear, formatted message
            changes_lines = []
            for grade in ['A', 'B', 'C', 'D', 'F']:
                count = changed_counts_readable.get(grade, 0)
                changes_lines.append(f"{grade:<2} : {count} record(s) changed")
            changes_text = "\n".join(changes_lines)

            self.status_label.config(
                text=(
                    f"✅ Data Correction Completed!\n\n"
                    f"Total incorrect records corrected : {incorrect_count}\n"
                    f"Total records after correction   : {len(df)}\n\n"
                    f"Changes by grade:\n{changes_text}"
                ),
                foreground="#228B22",
                justify=tk.LEFT
            )

        except Exception as e:
            self.status_label.config(text=f"❌ Error: {e}", foreground="red")




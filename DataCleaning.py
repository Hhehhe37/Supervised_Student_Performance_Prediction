import pandas as pd
import tkinter as tk
import os
from tkinter import messagebox

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class DataCleaningWindow:
    def __init__(self, window): 
        self.root = window
        self.root.title("Student Performance Missing Value Checker")
        self.root.geometry("1280x800")
        self.df = None

        self.file_path = os.path.join(OUTPUT_FOLDER, "Student_performance_data _.csv")

        # GUI Elements
        self.label = tk.Label(self.root, text="Missing Value Report", font=("Arial", 14))
        self.label.pack(pady=10)

        self.output_text = tk.Text(self.root, height=25, width=100)
        self.output_text.pack(pady=10)

        # Remove incorrect data first
        self.remove_incorrect_data()

        # Then check missing values
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
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            return

        # Calculate expected GradeClass from GPA
        df['ExpectedGrade'] = df['GPA'].apply(self.assign_grade_class)

        # Filter rows where GradeClass matches ExpectedGrade
        correct_df = df[df['GradeClass'] == df['ExpectedGrade']].copy()

        # Update self.df and save cleaned data
        self.df = correct_df.drop(columns=['ExpectedGrade'])
        self.df.to_csv(os.path.join(OUTPUT_FOLDER, "Corrected_StudentsPerformance.csv"), index=False)

        self.output_text.config(state="normal")   
        self.output_text.insert(tk.END, f"Removed incorrect records based on GPA and GradeClass mismatch.\n")
        self.output_text.insert(tk.END, f"Records before: {len(df)}, after: {len(self.df)}\n")
        self.output_text.insert(tk.END, "Cleaned data saved as 'Corrected_StudentsPerformance.csv'\n")
        self.output_text.config(state="disabled")


    def check_missing(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")
                return
        
        self.output_text.config(state="normal") 
        self.output_text.insert(tk.END, f"\nChecking missing values...\n")

        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()

        self.output_text.insert(tk.END, "Missing values per column:\n")
        self.output_text.insert(tk.END, f"{missing_counts}\n")

        if total_missing > 0:
            
            self.output_text.insert(tk.END, f"\nTotal missing values: {total_missing}\n")
            self.output_text.insert(tk.END, "Cleaning data...\n")
            self.df = self.df.dropna()
            self.output_text.insert(tk.END, f"New shape after cleaning: {self.df.shape}\n")

            # Save cleaned data
            self.df.to_csv(os.path.join(OUTPUT_FOLDER, "Cleaned_StudentsPerformance.csv"), index=False)
            self.output_text.insert(tk.END, "Cleaned data saved as 'Cleaned_StudentsPerformance.csv'\n")
        else:
            self.output_text.insert(tk.END, "No missing values found.\n")

        self.output_text.config(state="disabled") 
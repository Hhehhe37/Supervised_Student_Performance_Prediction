import pandas as pd
import tkinter as tk
from tkinter import ttk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DataPreprocessing.Process_incorrect_data import ReplaceIncorrectDataWindow


class CheckDuplicateWindow:
    def __init__(self, window): 
        """
        Initialize the CheckDuplicateWindow class.
        Args:
            window: The tkinter window to be used as the root window.
        """
        self.root = window
        self.root.title("🔁 Duplicate Record Checker")  # Set window title with emoji
        self.root.geometry("950x700")  # Set window dimensions
        self.root.configure(bg="#f5f7fa")  # Set background color

        # Define file path for the CSV data
        self.file_path = os.path.join("CSV_Files", "Student_performance_data _.csv")

        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create title label with font styling
        title_label = ttk.Label(
            main_frame, text="Duplicate Records Checker", 
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(pady=(0, 15))

        # Frame for table (only shown if duplicates exist)
        self.tree_frame = ttk.Frame(main_frame)

        # Create Treeview widget for displaying duplicates
        self.tree = ttk.Treeview(self.tree_frame, show="headings")
        # Create scrollbar for the Treeview
        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Status message label for displaying information
        self.status_label = ttk.Label(
            main_frame, text="", font=("Segoe UI", 12), foreground="#333"
        )
        self.status_label.pack(pady=10)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        remove_button = ttk.Button(button_frame, text="🗑 Remove Duplicates", command=self.remove_duplicates)
        remove_button.grid(row=0, column=0, padx=10)

        next_button = ttk.Button(button_frame, text="Next ➡️ Remove Incorrect Data", command=self.open_remove_incorrect)
        next_button.grid(row=0, column=1, padx=10)

        # Placeholder for chart
        self.chart_canvas = None

        self.load_and_check_duplicates()

    def load_and_check_duplicates(self):
        try:
            df = pd.read_csv(self.file_path)
            duplicate_df = df[df.duplicated()]

            total_records = len(df)
            duplicate_count = len(duplicate_df)
            percentage = (duplicate_count / total_records * 100) if total_records > 0 else 0

            if not duplicate_df.empty:
                # Show Treeview
                self.tree_frame.pack(fill=tk.BOTH, expand=True)
                self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                self.tree["columns"] = list(duplicate_df.columns)
                for col in duplicate_df.columns:
                    self.tree.heading(col, text=col)
                    self.tree.column(col, width=120)

                # Highlight duplicates with tag
                self.tree.tag_configure("duplicate", background="#ffe5e5")

                for _, row in duplicate_df.iterrows():
                    self.tree.insert("", tk.END, values=list(row), tags=("duplicate",))

                self.status_label.config(
                    text=f"Total Records: {total_records} | "
                         f"Duplicate Records: {duplicate_count} ({percentage:.2f}%)"
                )

                # Show pie chart
                self.show_chart(duplicate_count, total_records - duplicate_count)

            else:
                self.status_label.config(
                    text=f"Total Records: {total_records} | No duplicate records found."
                )

        except Exception as e:
            self.status_label.config(text=f"Error loading file:\n{e}")

    def show_chart(self, duplicate_count, unique_count):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            [duplicate_count, unique_count],
            labels=["Duplicates", "Unique"],
            autopct="%1.1f%%",
            colors=["#ff9999", "#99ff99"],
            startangle=90
        )
        ax.set_title("Duplicate Records Distribution")

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(pady=15)

    def remove_duplicates(self):
        """
        Removes duplicate rows from the CSV file and updates the file.
        Also clears the current table and chart display.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(self.file_path)
            # Remove duplicate rows from the DataFrame
            cleaned_df = df.drop_duplicates()
            # Save the cleaned DataFrame back to the CSV file
            cleaned_df.to_csv(self.file_path, index=False)
            # Update status label to indicate successful operation
            self.status_label.config(text="✅ Duplicates removed and file updated!")

            # Clear table & chart
            for i in self.tree.get_children():
                self.tree.delete(i)
            if self.chart_canvas:
                self.chart_canvas.get_tk_widget().destroy()

        except Exception as e:
            self.status_label.config(text=f"Error removing duplicates:\n{e}")

    def open_remove_incorrect(self):

        """
        This method opens a new window for replacing incorrect data.
        It destroys the current root window and creates a new Tk window,
        then initializes the ReplaceIncorrectDataWindow with the new root.
        """
        self.root.destroy()  # Close the current window
        new_root = tk.Tk()
        ReplaceIncorrectDataWindow(new_root)
        new_root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = CheckDuplicateWindow(root)
    root.mainloop()

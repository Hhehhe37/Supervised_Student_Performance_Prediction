import pandas as pd
import tkinter as tk
from tkinter import messagebox
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

OUTPUT_FOLDER = "CSV_Files"

class KNNPerformanceWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Performance Measure")
        self.root.geometry("1280x800")

        self.label = tk.Label(root, text="Enter K values to test: ", font=("Arial",12))
        self.label.pack(pady=10)

        self.k_entry = tk.Entry(root, font=("Arial", 12), width=30)
        self.k_entry.pack(pady=5)

        self.text_button = tk.Button(root, text="Evaluate", font=("Arial", 12), command=self.evaluate_knn)
        self.text_button.pack(pady=10)

        self.output_text = tk.Text(root, height=20, width=80)
        self.output_text.pack(pady=10)
        self.output_text.config(state="disabled")

    def evaluate_knn(self):
        try:

            k = int(self.k_entry.get())

            self.df = None
            self.file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(self.file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"k = {k}, Accuracy = {accuracy:.2f}%")
            self.output_text.config(state="disabled")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid integer for k.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
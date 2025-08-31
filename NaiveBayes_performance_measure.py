import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

OUTPUT_FOLDER = "CSV_Files"

class NaiveBayesPerformanceWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Naive Bayes Performance Evaluation")
        self.root.geometry("950x700")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🔍 Naive Bayes Classifier Evaluation", 
                               font=("Segoe UI", 20, "bold"), bg="#f5f7fa", fg="#222")
        title_label.pack(pady=15)

        # Run Evaluation Button
        self.text_button = ttk.Button(root, text="Run Evaluation", command=self.evaluate_nb)
        self.text_button.pack(pady=10)

        # Notebook (Tabs)
        notebook = ttk.Notebook(root)
        notebook.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Tab 1: Accuracy Results
        self.tab1 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab1, text="📊 Accuracy Results")

        # Table for results
        self.tree = ttk.Treeview(self.tab1, columns=("classifier", "accuracy"), show="headings", height=15)
        self.tree.heading("classifier", text="Classifier")
        self.tree.heading("accuracy", text="Accuracy (%)")
        self.tree.column("classifier", anchor="center", width=150)
        self.tree.column("accuracy", anchor="center", width=150)
        self.tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Label for best result
        self.best_label = tk.Label(self.tab1, text="", font=("Segoe UI", 12, "bold"), 
                                   bg="#ffffff", fg="#0a84ff", anchor="w", justify="left")
        self.best_label.pack(pady=10, padx=10, anchor="w")

        # Tab 2: Confusion Matrix & Report
        self.tab2 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab2, text="📑 Detailed Report")

        # Scrollable Text
        self.output_text = tk.Text(self.tab2, height=25, font=("Courier New", 11), wrap="none")
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_text.config(state="disabled")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.tab2, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def evaluate_nb(self):
        try:
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Standardize features (optional for Naive Bayes)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Train Naive Bayes
            nb = GaussianNB()
            nb.fit(x_train, y_train)
            y_pred = nb.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

            # Clear Treeview
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Insert result
            self.tree.insert("", "end", values=("Naive Bayes", f"{accuracy:.2f}"))

            # Best label
            self.best_label.config(
                text=f"✅ Classifier: Naive Bayes\n🎯 Accuracy: {accuracy:.2f}%"
            )

            # Confusion Matrix & Classification Report
            cm = confusion_matrix(y_test, y_pred)
            cm_accuracy = cm.trace() / cm.sum() * 100
            report = classification_report(y_test, y_pred, digits=2)

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            self.output_text.insert(tk.END, "📊 Confusion Matrix:\n")
            self.output_text.insert(tk.END, f"{cm}\n")
            self.output_text.insert(tk.END, f"\nCalculated Accuracy from CM: {cm_accuracy:.2f}%\n\n")
            self.output_text.insert(tk.END, "📈 Classification Report (Precision, Recall, F1-Score):\n")
            self.output_text.insert(tk.END, f"{report}")

            self.output_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NaiveBayesPerformanceWindow(root)
    root.mainloop()

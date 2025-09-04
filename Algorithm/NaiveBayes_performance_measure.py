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
    def __init__(self, root, callback=None):
        self.root = root
        self.callback = callback  # Add callback for returning results
        self.best_results = None  # Store results
        
        self.root.title("📈 Naive Bayes Performance Evaluation")
        self.root.geometry("950x750")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🔍 Naive Bayes Classifier Evaluation", 
                               font=("Segoe UI", 20, "bold"), bg="#f5f7fa", fg="#222")
        title_label.pack(pady=15)

        # Controls Frame
        controls_frame = tk.Frame(root, bg="#f5f7fa")
        controls_frame.pack(pady=10)

        # Train-Test Split Ratio Selection
        ratio_label = tk.Label(controls_frame, text="Train-Test Split Ratio:", font=("Segoe UI", 12, "bold"), bg="#f5f7fa", fg="#333")
        ratio_label.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="e")

        self.ratio_var = tk.StringVar()
        self.ratio_combo = ttk.Combobox(controls_frame, textvariable=self.ratio_var, 
                                       values=["60:40", "70:30", "80:20", "90:10"], 
                                       state="readonly", width=10)
        self.ratio_combo.set("80:20")  # Default value
        self.ratio_combo.grid(row=0, column=1, padx=(0, 20), pady=5)

        # Run Evaluation Button
        self.text_button = ttk.Button(controls_frame, text="Run Evaluation", command=self.evaluate_nb)
        self.text_button.grid(row=0, column=2, padx=10, pady=5)

        # Notebook (Tabs)
        notebook = ttk.Notebook(root)
        notebook.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Tab 1: Accuracy Results
        self.tab1 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab1, text="📊 Accuracy Results")

        # Configuration info frame
        config_frame = tk.Frame(self.tab1, bg="#ffffff")
        config_frame.pack(pady=10, padx=10, fill="x")

        self.config_label = tk.Label(config_frame, text="", font=("Segoe UI", 10), 
                                     bg="#ffffff", fg="#666", anchor="w", justify="left")
        self.config_label.pack(anchor="w")

        # Table for results
        self.tree = ttk.Treeview(self.tab1, columns=("classifier", "accuracy"), show="headings", height=12)
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

    def get_test_size_from_ratio(self, ratio_string):
        """Convert ratio string like '80:20' to test_size float like 0.2"""
        train_ratio, test_ratio = map(int, ratio_string.split(':'))
        return test_ratio / (train_ratio + test_ratio)

    def evaluate_nb(self):
        try:
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            # Get selected ratio and convert to test_size
            selected_ratio = self.ratio_var.get()
            test_size = self.get_test_size_from_ratio(selected_ratio)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

            # Update configuration display
            self.config_label.config(
                text=f"🔧 Configuration: Train-Test Split = {selected_ratio} | Training samples: {len(x_train)} | Testing samples: {len(x_test)}"
            )

            # Standardize features (optional for Naive Bayes)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Train Naive Bayes
            nb = GaussianNB()
            nb.fit(x_train, y_train)
            y_pred = nb.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100

            # Store results for comparison
            self.best_results = {
                'model': 'Naive Bayes',
                'accuracy': accuracy,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'ratio': selected_ratio,
                'train_samples': len(x_train),
                'test_samples': len(x_test)
            }

            # Clear Treeview
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Insert result
            self.tree.insert("", "end", values=("Naive Bayes", f"{accuracy:.2f}"))

            # Best label
            self.best_label.config(
                text=f"✅ Classifier: Naive Bayes\n🎯 Accuracy: {accuracy:.2f}%\n📊 Split Ratio: {selected_ratio}"
            )

            # Confusion Matrix & Classification Report
            cm = confusion_matrix(y_test, y_pred)
            cm_accuracy = cm.trace() / cm.sum() * 100
            report = classification_report(y_test, y_pred, digits=2)

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            # Show configuration info in detailed report
            self.output_text.insert(tk.END, f"🔧 Configuration:\n")
            self.output_text.insert(tk.END, f"   Train-Test Split: {selected_ratio}\n")
            self.output_text.insert(tk.END, f"   Training samples: {len(x_train)}\n")
            self.output_text.insert(tk.END, f"   Testing samples: {len(x_test)}\n\n")

            self.output_text.insert(tk.END, "📊 Confusion Matrix:\n")
            self.output_text.insert(tk.END, f"{cm}\n")
            self.output_text.insert(tk.END, f"\nCalculated Accuracy from CM: {cm_accuracy:.2f}%\n\n")
            self.output_text.insert(tk.END, "📈 Classification Report (Precision, Recall, F1-Score):\n")
            self.output_text.insert(tk.END, f"{report}")

            self.output_text.config(state="disabled")
            
            # Call callback if provided
            if self.callback:
                self.callback('Naive Bayes', self.best_results)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_best_results(self):
        """Return the results for comparison"""
        return self.best_results


if __name__ == "__main__":
    root = tk.Tk()
    app = NaiveBayesPerformanceWindow(root)
    root.mainloop()
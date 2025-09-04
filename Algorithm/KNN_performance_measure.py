import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

OUTPUT_FOLDER = "CSV_Files"

class KNNPerformanceWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 KNN Performance Evaluation")
        self.root.geometry("950x750")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🔍 KNN Classifier - Find Best k (1 to 20)", font=("Segoe UI", 18, "bold"), bg="#f5f7fa", fg="#333")
        title_label.pack(pady=20)

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
        self.text_button = ttk.Button(controls_frame, text="Run Evaluation", command=self.evaluate_knn)
        self.text_button.grid(row=0, column=2, padx=10, pady=5)

        # Output Frame with Label
        output_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        output_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        output_label = tk.Label(output_frame, text="Evaluation Results", font=("Segoe UI", 14, "bold"), bg="#ffffff", anchor="w")
        output_label.pack(anchor="w", padx=10, pady=10)

        # Scrollable Text Widget
        self.output_text = tk.Text(output_frame, height=20, font=("Courier New", 11), wrap="none")
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_text.config(state="disabled")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def get_test_size_from_ratio(self, ratio_string):
        """Convert ratio string like '80:20' to test_size float like 0.2"""
        train_ratio, test_ratio = map(int, ratio_string.split(':'))
        return test_ratio / (train_ratio + test_ratio)

    def evaluate_knn(self):
        try:
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            # Get selected ratio and convert to test_size
            selected_ratio = self.ratio_var.get()
            test_size = self.get_test_size_from_ratio(selected_ratio)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            best_k = None
            best_accuracy = 0
            results = []

            for k in range(1, 21):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred) * 100
                results.append((k, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k
                    best_y_pred = y_pred  # save predictions for best k

            # Display results
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            # Show configuration info
            self.output_text.insert(tk.END, f"🔧 Configuration:\n")
            self.output_text.insert(tk.END, f"   Train-Test Split: {selected_ratio}\n")
            self.output_text.insert(tk.END, f"   Training samples: {len(x_train)}\n")
            self.output_text.insert(tk.END, f"   Testing samples: {len(x_test)}\n\n")

            self.output_text.insert(tk.END, f"{'K Value':<10} {'Accuracy (%)':>15}\n")
            self.output_text.insert(tk.END, "-" * 30 + "\n")
            for k, acc in results:
                self.output_text.insert(tk.END, f"{k:<10} {acc:>15.2f}\n")

            self.output_text.insert(tk.END, "\n")
            self.output_text.insert(tk.END, f"✅ Best k value: {best_k}\n")
            self.output_text.insert(tk.END, f"🎯 Highest accuracy: {best_accuracy:.2f}%\n\n")

            # Confusion Matrix
            cm = confusion_matrix(y_test, best_y_pred)
            self.output_text.insert(tk.END, "📊 Confusion Matrix:\n")
            self.output_text.insert(tk.END, f"{cm}\n\n")

            # Precision, Recall, F1-score
            report = classification_report(y_test, best_y_pred, digits=2)
            self.output_text.insert(tk.END, "📈 Classification Report (Precision, Recall, F1-Score):\n")
            self.output_text.insert(tk.END, f"{report}")

            self.output_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNPerformanceWindow(root)
    root.mainloop()
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

OUTPUT_FOLDER = "CSV_Files"

class KNNPerformanceWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 KNN Performance Evaluation")
        self.root.geometry("850x650")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🔍 KNN Classifier - Find Best k (1 to 20)", font=("Segoe UI", 18, "bold"), bg="#f5f7fa", fg="#333")
        title_label.pack(pady=20)

        # Run Evaluation Button
        self.text_button = ttk.Button(root, text="Run Evaluation", command=self.evaluate_knn)
        self.text_button.pack(pady=10)

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

    def evaluate_knn(self):
        try:
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # ✅ Fix: Scale x_test using transform, not fit_transform
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            best_k = None
            best_accuracy = 0
            results = []

            for k in range(1, 21):  # Testing k from 1 to 20
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred) * 100
                results.append((k, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k

            # Display results
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            self.output_text.insert(tk.END, f"{'K Value':<10} {'Accuracy (%)':>15}\n")
            self.output_text.insert(tk.END, "-" * 30 + "\n")
            for k, acc in results:
                self.output_text.insert(tk.END, f"{k:<10} {acc:>15.2f}\n")

            self.output_text.insert(tk.END, "\n")
            self.output_text.insert(tk.END, f"✅ Best k value: {best_k}\n")
            self.output_text.insert(tk.END, f"🎯 Highest accuracy: {best_accuracy:.2f}%")

            self.output_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNPerformanceWindow(root)
    root.mainloop()

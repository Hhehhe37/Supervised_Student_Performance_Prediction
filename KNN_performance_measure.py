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
        self.root.title("KNN Performance Measure - Auto Best K")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Automatically finding best k from 1 to 20...", font=("Arial", 14))
        self.label.pack(pady=20)

        self.text_button = tk.Button(root, text="Run Evaluation", font=("Arial", 12), command=self.evaluate_knn)
        self.text_button.pack(pady=10)

        self.output_text = tk.Text(root, height=20, width=80)
        self.output_text.pack(pady=10)
        self.output_text.config(state="disabled")

    def evaluate_knn(self):
        try:
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            x = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            # Uncomment below to show all k accuracies:
            # for k, acc in results:
            #     self.output_text.insert(tk.END, f"k = {k}, Accuracy = {acc:.2f}%\n")
            self.output_text.insert(tk.END, f"Best k value: {best_k}\nHighest accuracy: {best_accuracy:.2f}%")
            self.output_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = KNNPerformanceWindow(root)
    root.mainloop()

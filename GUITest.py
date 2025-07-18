import tkinter as tk
from tkinter import ttk
from correlation_matrix import CorrelationMatrixView
from DataCleaning import DataCleaningWindow
from KNN_performance_measure import KNNPerformanceWindow
from DecisionTree_performance_measure import DecisionTreePerformanceWindow

class MissingValueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f0f4f7")

        # Create a main frame to center content
        main_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=500)

        # Title
        self.label = tk.Label(main_frame, text="🎓 Student Performance Prediction App", font=("Helvetica", 18, "bold"), bg="#ffffff", fg="#333")
        self.label.pack(pady=30)

        # Style buttons
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12), padding=10)
        style.map("TButton", background=[("active", "#0052cc")], foreground=[("active", "white")])

        button_width = 40

        # Buttons
        ttk.Button(main_frame, text="🧹 Data Cleaning", width=button_width, command=self.open_cleaning_window).pack(pady=10)
        ttk.Button(main_frame, text="📊 Correlation Matrix", width=button_width, command=self.open_correlation_window).pack(pady=10)
        ttk.Button(main_frame, text="🤖 KNN Performance", width=button_width, command=self.open_knn_prediction_window).pack(pady=10)
        ttk.Button(main_frame, text="🌳 Decision Tree Performance", width=button_width, command=self.open_dtree_prediction_window).pack(pady=10)
        ttk.Button(main_frame, text="📈 Student Performance Prediction", width=button_width).pack(pady=10)  # Future feature

    def open_cleaning_window(self):
        new_window = tk.Toplevel(self.root)
        DataCleaningWindow(new_window)

    def open_correlation_window(self):
        corr_view = CorrelationMatrixView()
        corr_view.show_correlation()

    def open_knn_prediction_window(self):
        knn_window = tk.Toplevel(self.root)
        KNNPerformanceWindow(knn_window)

    def open_dtree_prediction_window(self):
        dtree_window = tk.Toplevel(self.root)
        DecisionTreePerformanceWindow(dtree_window)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MissingValueApp(root)
    root.mainloop()

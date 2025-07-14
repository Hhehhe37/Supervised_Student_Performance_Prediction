import pandas as pd
import tkinter as tk
from tkinter import messagebox
from correlation_matrix import CorrelationMatrixView
from DataCleaning import DataCleaningWindow
from KNN_performance_measure import KNNPerformanceWindow
from DecisionTree_performance_measure import DecisionTreePerformanceWindow

class MissingValueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Menu")
        self.root.geometry("1280x800")

        # Menu Label
        self.label = tk.Label(root, text="Welcome to Student Performance Prediction App", font=("Arial", 16))
        self.label.pack(pady=20)

        button_width = 30

        self.cleaning_button = tk.Button(root, text="Data cleaning", font=("Arial", 14), width=button_width, command=self.open_cleaning_window)
        self.cleaning_button.pack(pady=10)

        self.correlation_button = tk.Button(root, text="Correlation Matrix", font=("Arial", 14), width=button_width, command=self.open_correlation_window)
        self.correlation_button.pack(pady=20)

        self.performance_button = tk.Button(root, text="KNN performance", font=("Arial", 14), width=button_width, command=self.open_knn_prediction_window)
        self.performance_button.pack(pady=20)

        self.dtree_button = tk.Button(root, text="Decision Tree performance", font=("Arial", 14), width=button_width, command=self.open_dtree_prediction_window)
        self.dtree_button.pack(pady=20)


        self.prediction_button = tk.Button(root, text="Student Performance Prediction", font=("Arial", 14), width=button_width) # !!! Need to add command to call function at future
        self.prediction_button.pack(pady=20)

    def open_cleaning_window(self):
        new_window = tk.Toplevel(self.root)
        DataCleaningWindow(new_window)

    def open_correlation_window(self):
        corr_view= CorrelationMatrixView()
        corr_view.show_correlation()

    def open_knn_prediction_window(self):
        knn_accuracy_window = tk.Toplevel(self.root)
        KNNPerformanceWindow(knn_accuracy_window)

    def open_dtree_prediction_window(self):
        dtree_accuracy_window = tk.Toplevel(self.root)
        DecisionTreePerformanceWindow(dtree_accuracy_window)
    
# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MissingValueApp(root)
    root.mainloop()

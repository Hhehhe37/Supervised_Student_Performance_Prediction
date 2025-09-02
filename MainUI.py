import tkinter as tk
from tkinter import ttk
from DataPreprocessing.correlation_matrix import CorrelationMatrixView
from DataPreprocessing.Check_Missing import CheckMissingWindow
from Algorithm.KNN_performance_measure import KNNPerformanceWindow
from Algorithm.DecisionTree_performance_measure import DecisionTreePerformanceWindow
from Algorithm.NaiveBayes_performance_measure import NaiveBayesPerformanceWindow
from Algorithm.SVM_performance_measure import SVMPerformanceWindow
from DataPreprocessing.RFE import RFEFeatureSelector

class MissingValueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction")
        self.root.geometry("1280x800")
        self.root.configure(bg="#dceeff")

        # Create a main frame to center content
        main_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=500)

        # Title
        self.label = tk.Label(main_frame, text="🎓 Student Performance Prediction App", font=("Helvetica", 18, "bold"), bg="#ffffff", fg="#333")
        self.label.pack(pady=20)

        # Create a canvas and scrollbar for scrollable content
        canvas_frame = tk.Frame(main_frame, bg="#ffffff")
        canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Canvas for scrolling
        self.canvas = tk.Canvas(canvas_frame, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#ffffff")

        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)

        # Button style
        button_style = {
            "bg": "#4fa4ff",        # Blue
            "fg": "white",          # White text
            "font": ("Segoe UI", 12),
            "width": 40,
            "height": 2,
            "bd": 0,
            "activebackground": "#0056b3",  # Darker blue on hover
            "activeforeground": "white",
        }

        # Create buttons in the scrollable frame
        self.create_buttons(button_style)

        # Center the scrollable content
        self.center_scrollable_content()

    def create_buttons(self, button_style):
        """Create all buttons in the scrollable frame"""
        buttons = [
            ("🧹 Data Preprocessing", self.open_cleaning_window),
            ("📊 Correlation Matrix", self.open_correlation_window),
            ("📊 RFE", self.open_rfe_window),
            ("🤖 KNN Performance", self.open_knn_prediction_window),
            ("🤖 Naive Bayes Performance", self.open_nb_prediction_window),
            ("🌳 Decision Tree Performance", self.open_dtree_prediction_window),
            ("📊 Model Comparison", self.open_model_comparison_window),
            ("📈 Student Performance Prediction", self.open_prediction_window),  # Future feature
        ]

        for text, command in buttons:
            btn = tk.Button(self.scrollable_frame, text=text, command=command, **button_style)
            btn.pack(pady=8)

    def center_scrollable_content(self):
        """Center the content within the scrollable frame"""
        self.scrollable_frame.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        frame_width = self.scrollable_frame.winfo_reqwidth()
        
        if canvas_width > frame_width:
            x_offset = (canvas_width - frame_width) // 2
            self.canvas.create_window((x_offset, 0), window=self.scrollable_frame, anchor="nw")

    def open_cleaning_window(self):
        new_window = tk.Toplevel(self.root)
        CheckMissingWindow(new_window)

    def open_correlation_window(self):
        corr_view = CorrelationMatrixView()
        corr_view.show_correlation()
        
    def open_rfe_window(self):
        rfe_view = RFEFeatureSelector()
        rfe_view.run_rfe()

    def open_knn_prediction_window(self):
        knn_window = tk.Toplevel(self.root)
        KNNPerformanceWindow(knn_window)
        
    def open_nb_prediction_window(self):
        nb_window = tk.Toplevel(self.root)
        NaiveBayesPerformanceWindow(nb_window)

    def open_dtree_prediction_window(self):
        dtree_window = tk.Toplevel(self.root)
        DecisionTreePerformanceWindow(dtree_window)

    def open_svm_prediction_window(self):
        svm_window = tk.Toplevel(self.root)
        SVMPerformanceWindow(svm_window)

    # Placeholder methods for additional buttons
    def open_model_comparison_window(self):
        print("Model Comparison window - To be implemented")

    def open_prediction_window(self):
        print("Student Performance Prediction window - To be implemented")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MissingValueApp(root)
    root.mainloop()
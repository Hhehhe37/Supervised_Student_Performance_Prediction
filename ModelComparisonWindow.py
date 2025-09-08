import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

OUTPUT_FOLDER = "CSV_Files"

class ModelComparisonWindow:
    """
    A comprehensive GUI window for comparing multiple machine learning models.
    Provides detailed analysis, visualization, and metrics comparison.
    """
    def __init__(self, root):
        """
        Initialize the ModelComparisonWindow with all UI components and settings.
        Args:
            root: The main tkinter window
        """
        self.root = root
        self.root.title("📊 Advanced Model Performance Comparison")
        self.root.geometry("1500x1000")
        self.root.configure(bg="#f5f7fa")

        self.results = {}  # Store all model results
        self.best_model_info = {}  # Store best model for each metric

        # Title
        title_label = tk.Label(root, text="🏆 Advanced Model Performance Comparison", 
                               font=("Segoe UI", 20, "bold"), bg="#f5f7fa", fg="#222")
        title_label.pack(pady=15)

        # Configuration Frame
        config_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        config_frame.pack(pady=10, padx=20, fill=tk.X)
        
        config_label = tk.Label(config_frame, text="Configuration Settings", font=("Segoe UI", 14, "bold"), 
                               bg="#ffffff", fg="#333")
        config_label.pack(anchor="w", padx=10, pady=5)

        # Train-Test Split Configuration
        split_frame = tk.Frame(config_frame, bg="#ffffff")
        split_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(split_frame, text="Train-Test Split:", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.split_var = tk.StringVar(value="80:20")
        split_options = ["60:40", "70:30", "80:20", "90:10"]
        self.split_combo = ttk.Combobox(split_frame, textvariable=self.split_var, 
                                       values=split_options, state="readonly", width=8)
        self.split_combo.pack(side=tk.LEFT, padx=10)
        
        # Add explanation label
        split_explain = tk.Label(split_frame, text="(Train:Test ratio)", 
                               font=("Segoe UI", 9), bg="#ffffff", fg="#666")
        split_explain.pack(side=tk.LEFT, padx=5)

        # Feature selection
        feature_frame = tk.Frame(config_frame, bg="#ffffff")
        feature_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(feature_frame, text="Number of top features (or 'all'):", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.top_k_entry = tk.Entry(feature_frame, font=("Segoe UI", 11), width=10)
        self.top_k_entry.pack(side=tk.LEFT, padx=10)
        self.top_k_entry.insert(0, "all")

        # Model selection checkboxes
        models_frame = tk.Frame(config_frame, bg="#ffffff")
        models_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(models_frame, text="Models to compare:", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.model_vars = {
            'KNN': tk.BooleanVar(value=True),
            'Naive Bayes': tk.BooleanVar(value=True),
            'Decision Tree': tk.BooleanVar(value=True),
            'SVM': tk.BooleanVar(value=True)
        }
        # Create checkboxes for each model
        for model_name, var in self.model_vars.items():
            cb = tk.Checkbutton(models_frame, text=model_name, variable=var, 
                               font=("Segoe UI", 10), bg="#ffffff")
            cb.pack(side=tk.LEFT, padx=10)

        # Run Comparison Button and Progress
        button_frame = tk.Frame(root, bg="#f5f7fa")
        button_frame.pack(pady=10)
        
        self.compare_button = ttk.Button(button_frame, text="🚀 Run Advanced Model Comparison", 
                                        command=self.run_comparison)
        self.compare_button.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(button_frame, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # Status label
        self.status_label = tk.Label(button_frame, text="Ready to run comparison", 
                                     font=("Segoe UI", 10), bg="#f5f7fa", fg="#666")
        self.status_label.pack(pady=2)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Tab 1: Comparison Table
        self.create_comparison_tab()
        
        # Tab 2: Best Model Analysis
        self.create_best_model_tab()
        
        # Tab 3: Detailed Metrics
        self.create_detailed_metrics_tab()
        
        # Tab 4: Visualization
        self.create_visualization_tab()

        # Tab 5: Training Performance
        self.create_performance_tab()

        # Tab 6: Statistics Measurement
        self.create_statistics_tab()

    # Additional methods for other tabs and functionality
    def get_test_size_from_split(self, split_ratio):
        """Convert split ratio string to test_size float"""
        train_pct, test_pct = split_ratio.split(":")
        test_size = int(test_pct) / (int(train_pct) + int(test_pct))
        return test_size

    # Placeholder methods for other tabs
    def create_comparison_tab(self):
        """Create the main comparison table tab"""
        self.tab1 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab1, text="📊 Model Comparison")

        # Comparison Table
        columns = ("Model", "Accuracy (%)", "Precision", "Recall", "F1-Score", 
                  "Best Parameter", "Train Time (s)", "Overall Rank")
        self.comparison_tree = ttk.Treeview(self.tab1, columns=columns, show="headings", height=10)
        
        column_widths = {
            "Model": 120, "Accuracy (%)": 100, "Precision": 100, "Recall": 100,
            "F1-Score": 100, "Best Parameter": 150, "Train Time (s)": 110, "Overall Rank": 120
        }
        
        for col in columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, anchor="center", width=column_widths.get(col, 100))

        self.comparison_tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar_comp = ttk.Scrollbar(self.tab1, orient="vertical", command=self.comparison_tree.yview)
        scrollbar_comp.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_tree.configure(yscrollcommand=scrollbar_comp.set)

        # Winner labels frame
        self.winners_frame = tk.Frame(self.tab1, bg="#ffffff")
        self.winners_frame.pack(pady=20, fill=tk.X)
        
        # Data split info
        self.split_info_label = tk.Label(self.winners_frame, text="", font=("Segoe UI", 12, "italic"), 
                                       bg="#ffffff", fg="#666")
        self.split_info_label.pack(pady=2)
        
        # Overall winner
        self.overall_winner_label = tk.Label(self.winners_frame, text="", font=("Segoe UI", 16, "bold"), 
                                           bg="#ffffff", fg="#228B22")
        self.overall_winner_label.pack(pady=5)

        # Individual metric winners
        self.metric_winners_label = tk.Label(self.winners_frame, text="", font=("Segoe UI", 12), 
                                           bg="#ffffff", fg="#0066cc", justify="left")
        self.metric_winners_label.pack(pady=5)

    def create_best_model_tab(self):
        """Create best model analysis tab"""
        self.tab2 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab2, text="🥇 Best Model Analysis")

        # Best model summary
        self.best_model_text = tk.Text(self.tab2, height=30, font=("Segoe UI", 11), wrap="word")
        scrollbar_best = ttk.Scrollbar(self.tab2, command=self.best_model_text.yview)
        self.best_model_text.config(yscrollcommand=scrollbar_best.set)
        
        self.best_model_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar_best.pack(side=tk.RIGHT, fill=tk.Y)
        self.best_model_text.config(state="disabled")

    def create_detailed_metrics_tab(self):
        """Create detailed metrics tab"""
        self.tab3 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab3, text="📑 Detailed Metrics")

        # Scrollable text for detailed results
        self.detailed_text = tk.Text(self.tab3, height=30, font=("Courier New", 10), wrap="word")
        scrollbar3 = ttk.Scrollbar(self.tab3, command=self.detailed_text.yview)
        self.detailed_text.config(yscrollcommand=scrollbar3.set)
        
        self.detailed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)
        self.detailed_text.config(state="disabled")

    def create_visualization_tab(self):
        """Create visualization tab"""
        self.tab4 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab4, text="📈 Visualization")

        # Create matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Advanced Model Performance Analysis', fontsize=16, fontweight='bold')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.tab4)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_performance_tab(self):
        """Create training performance analysis tab"""
        self.tab5 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab5, text="⚡ Training Performance")

        # Performance analysis text
        self.performance_text = tk.Text(self.tab5, height=30, font=("Courier New", 11), wrap="word")
        scrollbar_perf = ttk.Scrollbar(self.tab5, command=self.performance_text.yview)
        self.performance_text.config(yscrollcommand=scrollbar_perf.set)
        
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar_perf.pack(side=tk.RIGHT, fill=tk.Y)
        self.performance_text.config(state="disabled")

    def create_statistics_tab(self):
        """Create statistics tab"""
        self.tab6 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab6, text="📊 Statistics")

        # Statistics text
        self.statistics_text = tk.Text(self.tab6, height=30, font=("Courier New", 11), wrap="word")
        scrollbar_stats = ttk.Scrollbar(self.tab6, command=self.statistics_text.yview)
        self.statistics_text.config(yscrollcommand=scrollbar_stats.set)

        self.statistics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar_stats.pack(side=tk.RIGHT, fill=tk.Y)
        self.statistics_text.config(state="disabled")

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()

    def run_comparison(self):
        """Run comparison between selected models with configurable train-test split"""
        try:
            # Disable button during evaluation
            self.compare_button.config(state="disabled")
            
            # Get selected models
            selected_models = [model for model, var in self.model_vars.items() if var.get()]
            if not selected_models:
                messagebox.showwarning("Warning", "Please select at least one model to compare.")
                return

            total_models = len(selected_models)
            self.progress['maximum'] = total_models * 100

            # Load data
            self.update_status("Loading and preparing data...")
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            # Get train-test split ratio
            split_ratio = self.split_var.get()
            test_size = self.get_test_size_from_split(split_ratio)
            self.update_status(f"Using {split_ratio} train-test split...")

            # Feature selection
            user_input = self.top_k_entry.get().strip().lower()
            if user_input == "all" or user_input == "":
                k = None
            else:
                try:
                    k = int(user_input)
                except ValueError:
                    k = None

            # Calculate correlation and select features
            correlations = df.corr(numeric_only=True)["GradeClass"].abs().sort_values(ascending=False)
            correlations = correlations.drop(labels=["GradeClass"], errors='ignore')

            if k is not None and k > 0:
                top_features = correlations.head(k).index.tolist()
            else:
                top_features = correlations.index.tolist()

            top_features.append("GradeClass")
            df_filtered = df[top_features]

            X = df_filtered.drop("GradeClass", axis=1)
            y = df_filtered["GradeClass"]

            # Use configurable train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Store split info for display
            train_size = len(X_train)
            test_size_actual = len(X_test)
            self.split_info = {
                'ratio': split_ratio,
                'train_size': train_size,
                'test_size': test_size_actual,
                'total_size': len(df_filtered)
            }

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize results
            self.results = {}
            current_model = 0

            # Test selected models with optimizations
            if 'KNN' in selected_models:
                current_model += 1
                self.progress['value'] = (current_model - 1) * 100
                self.update_status(f"Optimizing KNN ({current_model}/{total_models})...")
                self.evaluate_knn_optimized(X_train_scaled, X_test_scaled, y_train, y_test)
                self.progress['value'] = current_model * 100

            if 'Naive Bayes' in selected_models:
                current_model += 1
                self.progress['value'] = (current_model - 1) * 100
                self.update_status(f"Evaluating Naive Bayes ({current_model}/{total_models})...")
                self.evaluate_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
                self.progress['value'] = current_model * 100

            if 'Decision Tree' in selected_models:
                current_model += 1
                self.progress['value'] = (current_model - 1) * 100
                self.update_status(f"Optimizing Decision Tree ({current_model}/{total_models})...")
                self.evaluate_decision_tree_optimized(X_train, X_test, y_train, y_test)
                self.progress['value'] = current_model * 100

            if 'SVM' in selected_models:
                current_model += 1
                self.progress['value'] = (current_model - 1) * 100
                self.update_status(f"Optimizing SVM ({current_model}/{total_models})...")
                self.evaluate_svm_optimized(X_train_scaled, X_test_scaled, y_train, y_test)
                self.progress['value'] = current_model * 100

            # Find best models
            self.update_status("Analyzing results...")
            self.find_best_models()

            # Update all displays
            self.update_comparison_table()
            self.update_best_model_analysis(len(top_features) - 1)
            self.update_detailed_metrics()
            self.update_visualization()
            self.update_performance_analysis()

            self.update_status(f"Comparison completed! Analyzed {len(selected_models)} models with {split_ratio} split.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            # Re-enable button
            self.compare_button.config(state="normal")
            self.progress['value'] = self.progress['maximum']

    # [Include all the evaluation methods from the original code here]
    # evaluate_knn_optimized, evaluate_naive_bayes, evaluate_decision_tree_optimized, evaluate_svm_optimized
    # find_best_models, calculate_model_rank, update_comparison_table, etc.
    
    def evaluate_knn_optimized(self, X_train, X_test, y_train, y_test):
        """Evaluate KNN with k optimization (1-20)"""
        best_k, best_accuracy = 1, 0
        best_y_pred = None
        best_train_time = 0

        for k in range(1, 21):
            start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_y_pred = y_pred
                best_train_time = train_time

        # Calculate all metrics for best k
        precision = precision_score(y_test, best_y_pred, average='weighted')
        recall = recall_score(y_test, best_y_pred, average='weighted')
        f1 = f1_score(y_test, best_y_pred, average='weighted')

        self.results['KNN'] = {
            'accuracy': best_accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': f"k={best_k}",
            'train_time': best_train_time,
            'y_pred': best_y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, best_y_pred),
            'classification_report': classification_report(y_test, best_y_pred),
            'model_specific': {'neighbors': best_k, 'total_tested': 20}
        }

    def evaluate_naive_bayes(self, X_train, X_test, y_train, y_test):
        """Evaluate Naive Bayes"""
        start_time = time.time()
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = nb.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        self.results['Naive Bayes'] = {
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': "Gaussian",
            'train_time': train_time,
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'model_specific': {'algorithm': 'Gaussian', 'assumptions': 'Features are independent'}
        }

    def evaluate_decision_tree_optimized(self, X_train, X_test, y_train, y_test):
        """Evaluate Decision Tree with hyperparameter optimization"""
        max_depths = [3, 5, 7, 10, 15, None]
        min_samples_splits = [2, 5, 10]
        min_samples_leafs = [1, 2, 5]

        best_accuracy = 0
        best_params = {}
        best_y_pred = None
        best_train_time = 0
        best_dt = None

        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    start_time = time.time()
                    dt = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    
                    dt.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    y_pred = dt.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }
                        best_y_pred = y_pred
                        best_train_time = train_time
                        best_dt = dt

        precision = precision_score(y_test, best_y_pred, average='weighted')
        recall = recall_score(y_test, best_y_pred, average='weighted')
        f1 = f1_score(y_test, best_y_pred, average='weighted')

        param_str = f"depth={best_params['max_depth']}, split={best_params['min_samples_split']}"

        self.results['Decision Tree'] = {
            'accuracy': best_accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': param_str,
            'train_time': best_train_time,
            'y_pred': best_y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, best_y_pred),
            'classification_report': classification_report(y_test, best_y_pred),
            'model_specific': {
                'params': best_params,
                'n_nodes': best_dt.tree_.node_count,
                'n_leaves': best_dt.tree_.n_leaves,
                'max_depth_actual': best_dt.tree_.max_depth
            }
        }

    def evaluate_svm_optimized(self, X_train, X_test, y_train, y_test):
        """Evaluate SVM with kernel and parameter optimization"""
        kernels = ['linear', 'rbf', 'poly']
        param_grids = {
            'linear': {'C': [0.1, 1, 10]},
            'rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01]},
            'poly': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'degree': [2, 3]}
        }

        best_accuracy = 0
        best_params = {}
        best_y_pred = None
        best_train_time = 0
        best_svm = None

        for kernel in kernels:
            if kernel == 'linear':
                for C in param_grids[kernel]['C']:
                    start_time = time.time()
                    svm = SVC(kernel=kernel, C=C, random_state=42)
                    svm.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    y_pred = svm.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'kernel': kernel, 'C': C}
                        best_y_pred = y_pred
                        best_train_time = train_time
                        best_svm = svm
            
            elif kernel == 'rbf':
                for C in param_grids[kernel]['C']:
                    for gamma in param_grids[kernel]['gamma']:
                        start_time = time.time()
                        svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                        svm.fit(X_train, y_train)
                        train_time = time.time() - start_time
                        
                        y_pred = svm.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
                            best_y_pred = y_pred
                            best_train_time = train_time
                            best_svm = svm
            
            elif kernel == 'poly':
                for C in param_grids[kernel]['C']:
                    for gamma in param_grids[kernel]['gamma']:
                        for degree in param_grids[kernel]['degree']:
                            start_time = time.time()
                            svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
                            svm.fit(X_train, y_train)
                            train_time = time.time() - start_time
                            
                            y_pred = svm.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree}
                                best_y_pred = y_pred
                                best_train_time = train_time
                                best_svm = svm

        precision = precision_score(y_test, best_y_pred, average='weighted')
        recall = recall_score(y_test, best_y_pred, average='weighted')
        f1 = f1_score(y_test, best_y_pred, average='weighted')

        # Create parameter string
        param_items = []
        for key, value in best_params.items():
            if key != 'kernel':
                param_items.append(f"{key}={value}")
        param_str = f"{best_params['kernel']} ({', '.join(param_items)})"

        self.results['SVM'] = {
            'accuracy': best_accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': param_str,
            'train_time': best_train_time,
            'y_pred': best_y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, best_y_pred),
            'classification_report': classification_report(y_test, best_y_pred),
            'model_specific': {
                'params': best_params,
                'support_vectors': best_svm.n_support_,
                'total_support_vectors': sum(best_svm.n_support_)
            }
        }

    def find_best_models(self):
        """Find the best model for each metric and overall"""
        if not self.results:
            return
        
        # Find best model for each metric
        self.best_model_info = {
            'accuracy': {'model': '', 'score': 0},
            'precision': {'model': '', 'score': 0},
            'recall': {'model': '', 'score': 0},
            'f1_score': {'model': '', 'score': 0},
            'speed': {'model': '', 'score': float('inf')},
            'overall': {'model': '', 'score': 0}
        }
        
        for model, metrics in self.results.items():
            # Check each metric
            if metrics['accuracy'] > self.best_model_info['accuracy']['score']:
                self.best_model_info['accuracy'] = {'model': model, 'score': metrics['accuracy']}
            
            if metrics['precision'] > self.best_model_info['precision']['score']:
                self.best_model_info['precision'] = {'model': model, 'score': metrics['precision']}
            
            if metrics['recall'] > self.best_model_info['recall']['score']:
                self.best_model_info['recall'] = {'model': model, 'score': metrics['recall']}
            
            if metrics['f1_score'] > self.best_model_info['f1_score']['score']:
                self.best_model_info['f1_score'] = {'model': model, 'score': metrics['f1_score']}
            
            if metrics['train_time'] < self.best_model_info['speed']['score']:
                self.best_model_info['speed'] = {'model': model, 'score': metrics['train_time']}
        
        # Calculate overall best (weighted average of all metrics)
        for model, metrics in self.results.items():
            # Normalize metrics for fair comparison
            normalized_accuracy = metrics['accuracy'] / 100
            
            # Speed score (inverse of time, normalized)
            max_time = max(self.results[m]['train_time'] for m in self.results)
            speed_score = (max_time - metrics['train_time']) / max_time if max_time > 0 else 0
            
            # Calculate weighted overall score
            overall_score = (normalized_accuracy * 0.40 + 
                           metrics['precision'] * 0.25 + 
                           metrics['recall'] * 0.25 + 
                           metrics['f1_score'] * 0.1)
            
            if overall_score > self.best_model_info['overall']['score']:
                self.best_model_info['overall'] = {'model': model, 'score': overall_score}

    def calculate_model_rank(self, model):
        """Calculate overall rank for a model based on all metrics"""
        metrics = self.results[model]
        
        # Count how many times this model is best in each metric
        best_count = 0
        if self.best_model_info['accuracy']['model'] == model:
            best_count += 1
        if self.best_model_info['precision']['model'] == model:
            best_count += 1
        if self.best_model_info['recall']['model'] == model:
            best_count += 1
        if self.best_model_info['f1_score']['model'] == model:
            best_count += 1
        if self.best_model_info['speed']['model'] == model:
            best_count += 1
        
        # Calculate average rank (lower is better)
        ranks = []
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metric_keys:
            sorted_models = sorted(self.results.keys(), 
                                 key=lambda x: self.results[x][metric], reverse=True)
            ranks.append(sorted_models.index(model) + 1)
        
        # Add speed rank (lower time = better rank)
        speed_sorted = sorted(self.results.keys(), 
                             key=lambda x: self.results[x]['train_time'])
        ranks.append(speed_sorted.index(model) + 1)
        
        avg_rank = sum(ranks) / len(ranks)
        return avg_rank, best_count

    def update_comparison_table(self):
        """Update the comparison table with split information"""
        # Clear existing data
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)

        # Update split info label
        if hasattr(self, 'split_info'):
            split_text = (f"Data Split: {self.split_info['ratio']} "
                         f"(Training: {self.split_info['train_size']} samples, "
                         f"Testing: {self.split_info['test_size']} samples)")
            self.split_info_label.config(text=split_text)

        # Insert results with rankings
        model_rankings = []
        for model, metrics in self.results.items():
            avg_rank, best_count = self.calculate_model_rank(model)
            model_rankings.append((model, avg_rank, best_count))
            
            self.comparison_tree.insert("", "end", values=(
                model,
                f"{metrics['accuracy']:.2f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                metrics['best_param'],
                f"{metrics['train_time']:.4f}",
                f"#{int(avg_rank)} ({best_count} wins)"
            ))

        # Update winner labels
        overall_winner = self.best_model_info['overall']['model']
        overall_score = self.best_model_info['overall']['score']
        self.overall_winner_label.config(
            text=f"🏆 Overall Best Model: {overall_winner} (Score: {overall_score:.3f})"
        )

        # Individual metric winners
        winners_text = (f"📊 Best Accuracy: {self.best_model_info['accuracy']['model']} "
                       f"({self.best_model_info['accuracy']['score']:.2f}%)\n"
                       f"🎯 Best Precision: {self.best_model_info['precision']['model']} "
                       f"({self.best_model_info['precision']['score']:.3f})\n"
                       f"📈 Best Recall: {self.best_model_info['recall']['model']} "
                       f"({self.best_model_info['recall']['score']:.3f})\n"
                       f"⚖️ Best F1-Score: {self.best_model_info['f1_score']['model']} "
                       f"({self.best_model_info['f1_score']['score']:.3f})\n"
                       f"⚡ Fastest Training: {self.best_model_info['speed']['model']} "
                       f"({self.best_model_info['speed']['score']:.4f}s)")
        self.metric_winners_label.config(text=winners_text)

    def update_best_model_analysis(self, num_features):
        """Update best model analysis with split information"""
        self.best_model_text.config(state="normal")
        self.best_model_text.delete("1.0", tk.END)

        if not self.results:
            self.best_model_text.insert(tk.END, "No results available. Please run the comparison first.")
            self.best_model_text.config(state="disabled")
            return

        best_model = self.best_model_info['overall']['model']
        best_metrics = self.results[best_model]

        analysis = f"""🏆 COMPREHENSIVE BEST MODEL ANALYSIS
{'='*80}

📊 DATASET CONFIGURATION:
Features Used: {num_features}
Data Split: {self.split_info['ratio']} (Train: {self.split_info['train_size']}, Test: {self.split_info['test_size']})
Total Samples: {self.split_info['total_size']}

🥇 OVERALL WINNER: {best_model}
Overall Performance Score: {self.best_model_info['overall']['score']:.4f}
Best Configuration: {best_metrics['best_param']}

📊 PERFORMANCE METRICS:
• Accuracy: {best_metrics['accuracy']:.2f}% 
• Precision: {best_metrics['precision']:.4f}
• Recall: {best_metrics['recall']:.4f}  
• F1-Score: {best_metrics['f1_score']:.4f}
• Training Time: {best_metrics['train_time']:.4f} seconds

🎯 WHY THIS MODEL IS THE OVERALL WINNER:
"""

        # Analyze why this model is best
        wins = []
        if self.best_model_info['accuracy']['model'] == best_model:
            wins.append("✅ Achieves highest accuracy")
        if self.best_model_info['precision']['model'] == best_model:
            wins.append("✅ Achieves highest precision")
        if self.best_model_info['recall']['model'] == best_model:
            wins.append("✅ Achieves highest recall")
        if self.best_model_info['f1_score']['model'] == best_model:
            wins.append("✅ Achieves highest F1-score")
        if self.best_model_info['speed']['model'] == best_model:
            wins.append("✅ Fastest training time")

        if wins:
            analysis += "\n".join(wins) + "\n\n"
        else:
            analysis += "• Provides the best balanced performance across all evaluation metrics\n\n"

        # Impact of train-test split
        split_ratio = self.split_info['ratio']
        train_pct = int(split_ratio.split(':')[0])
        test_pct = int(split_ratio.split(':')[1])
        
        analysis += f"📈 TRAIN-TEST SPLIT IMPACT ({split_ratio}):\n"
        if train_pct >= 85:
            analysis += f"• Large training set ({train_pct}%) - Better model learning capability\n"
            analysis += f"• Small test set ({test_pct}%) - Less robust evaluation, higher variance\n"
        elif train_pct >= 75:
            analysis += f"• Balanced split - Good compromise between training and evaluation\n"
        else:
            analysis += f"• Smaller training set ({train_pct}%) - May limit model complexity\n"
            analysis += f"• Large test set ({test_pct}%) - More robust evaluation, lower variance\n"
        analysis += "\n"

        # Model-specific insights
        analysis += f"🔍 MODEL-SPECIFIC INSIGHTS ({best_model}):\n"
        if best_model == 'KNN':
            specific = best_metrics['model_specific']
            analysis += f"• Optimal number of neighbors: {specific['neighbors']}\n"
            analysis += f"• Tested {specific['total_tested']} different k values\n"
            analysis += f"• Instance-based learning: Makes predictions based on similarity to training data\n"
            analysis += f"• Performance may vary with different train-test splits\n\n"
        
        elif best_model == 'Naive Bayes':
            specific = best_metrics['model_specific']
            analysis += f"• Algorithm type: {specific['algorithm']}\n"
            analysis += f"• Key assumption: {specific['assumptions']}\n"
            analysis += f"• Probabilistic model: Uses Bayes' theorem for predictions\n"
            analysis += f"• Generally stable across different train-test splits\n\n"
        
        elif best_model == 'Decision Tree':
            specific = best_metrics['model_specific']
            analysis += f"• Max depth: {specific['params']['max_depth']}\n"
            analysis += f"• Min samples to split: {specific['params']['min_samples_split']}\n"
            analysis += f"• Min samples per leaf: {specific['params']['min_samples_leaf']}\n"
            analysis += f"• Tree complexity: {specific['n_nodes']} nodes, {specific['n_leaves']} leaves\n"
            analysis += f"• May be sensitive to train-test split variations\n\n"
        
        elif best_model == 'SVM':
            specific = best_metrics['model_specific']
            analysis += f"• Kernel configuration: {specific['params']}\n"
            analysis += f"• Support vectors per class: {list(specific['support_vectors'])}\n"
            analysis += f"• Total support vectors: {specific['total_support_vectors']}\n"
            analysis += f"• Generally robust to different train-test splits\n\n"

        # Performance comparison
        analysis += f"📈 PERFORMANCE COMPARISON WITH OTHER MODELS:\n"
        analysis += "-" * 60 + "\n"
        
        # Sort models by overall performance
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: self.best_model_info['overall']['score'] 
                             if self.best_model_info['overall']['model'] == x[0] else 
                             (x[1]['accuracy']/100 * 0.35 + x[1]['precision'] * 0.25 + 
                              x[1]['recall'] * 0.25 + x[1]['f1_score'] * 0.2), 
                             reverse=True)

        for i, (model, metrics) in enumerate(sorted_models, 1):
            rank_indicator = "👑" if i == 1 else f"{i}."
            analysis += f"{rank_indicator} {model:<15}: "
            analysis += f"Acc={metrics['accuracy']:.1f}%, "
            analysis += f"Prec={metrics['precision']:.3f}, "
            analysis += f"Rec={metrics['recall']:.3f}, "
            analysis += f"F1={metrics['f1_score']:.3f}, "
            analysis += f"Time={metrics['train_time']:.4f}s\n"

        analysis += f"\n📋 DETAILED CLASSIFICATION REPORT:\n"
        analysis += "=" * 60 + "\n"
        analysis += f"{best_metrics['classification_report']}\n"

        analysis += f"📊 CONFUSION MATRIX:\n"
        analysis += f"{best_metrics['confusion_matrix']}\n\n"

        analysis += f"🎯 FINAL RECOMMENDATION:\n"
        analysis += "=" * 60 + "\n"
        analysis += f"Deploy {best_model} with configuration: {best_metrics['best_param']}\n"
        analysis += f"Using {split_ratio} train-test split for evaluation\n\n"
        
        analysis += f"This model provides:\n"
        if best_metrics['accuracy'] >= 90:
            analysis += f"• Excellent accuracy ({best_metrics['accuracy']:.1f}%) for reliable predictions\n"
        elif best_metrics['accuracy'] >= 80:
            analysis += f"• Good accuracy ({best_metrics['accuracy']:.1f}%) suitable for most applications\n"
        else:
            analysis += f"• Moderate accuracy ({best_metrics['accuracy']:.1f}%) - consider feature engineering\n"
        
        analysis += f"• Balanced precision and recall for reliable classification across all classes\n"
        analysis += f"• Training time of {best_metrics['train_time']:.4f}s suitable for your application needs\n"

        self.best_model_text.insert(tk.END, analysis)
        self.best_model_text.config(state="disabled")

    def update_detailed_metrics(self):
        """Update detailed metrics text"""
        self.detailed_text.config(state="normal")
        self.detailed_text.delete("1.0", tk.END)

        if not self.results:
            self.detailed_text.insert(tk.END, "No results available.")
            self.detailed_text.config(state="disabled")
            return

        # Add split information at the top
        if hasattr(self, 'split_info'):
            self.detailed_text.insert(tk.END, f"DATA CONFIGURATION\n")
            self.detailed_text.insert(tk.END, f"{'='*40}\n")
            self.detailed_text.insert(tk.END, f"Train-Test Split: {self.split_info['ratio']}\n")
            self.detailed_text.insert(tk.END, f"Training Samples: {self.split_info['train_size']}\n")
            self.detailed_text.insert(tk.END, f"Testing Samples: {self.split_info['test_size']}\n")
            self.detailed_text.insert(tk.END, f"Total Samples: {self.split_info['total_size']}\n\n")

        # Sort models by overall performance
        overall_scores = {}
        for model, metrics in self.results.items():
            if self.best_model_info['overall']['model'] == model:
                overall_scores[model] = self.best_model_info['overall']['score']
            else:
                # Calculate score for non-best models
                normalized_accuracy = metrics['accuracy'] / 100
                max_time = max(self.results[m]['train_time'] for m in self.results)
                speed_score = (max_time - metrics['train_time']) / max_time if max_time > 0 else 0
                overall_scores[model] = (normalized_accuracy * 0.35 + 
                                       metrics['precision'] * 0.25 + 
                                       metrics['recall'] * 0.25 + 
                                       metrics['f1_score'] * 0.1 + 
                                       speed_score * 0.05)

        sorted_models = sorted(self.results.keys(), 
                             key=lambda x: overall_scores[x], reverse=True)

        for i, model in enumerate(sorted_models, 1):
            metrics = self.results[model]
            avg_rank, best_count = self.calculate_model_rank(model)
            
            self.detailed_text.insert(tk.END, f"{'='*80}\n")
            self.detailed_text.insert(tk.END, f"#{i} - {model} COMPREHENSIVE ANALYSIS\n")
            self.detailed_text.insert(tk.END, f"{'='*80}\n\n")
            
            # Performance metrics
            self.detailed_text.insert(tk.END, f"📊 PERFORMANCE METRICS:\n")
            self.detailed_text.insert(tk.END, f"• Accuracy: {metrics['accuracy']:.2f}%\n")
            self.detailed_text.insert(tk.END, f"• Precision: {metrics['precision']:.4f}\n")
            self.detailed_text.insert(tk.END, f"• Recall: {metrics['recall']:.4f}\n")
            self.detailed_text.insert(tk.END, f"• F1-Score: {metrics['f1_score']:.4f}\n")
            self.detailed_text.insert(tk.END, f"• Training Time: {metrics['train_time']:.4f} seconds\n")
            self.detailed_text.insert(tk.END, f"• Overall Score: {overall_scores[model]:.4f}\n")
            self.detailed_text.insert(tk.END, f"• Average Rank: {avg_rank:.1f}\n")
            self.detailed_text.insert(tk.END, f"• Metrics Won: {best_count}/5\n\n")
            
            # Model configuration
            self.detailed_text.insert(tk.END, f"⚙️ OPTIMAL CONFIGURATION:\n")
            self.detailed_text.insert(tk.END, f"• Best Parameters: {metrics['best_param']}\n")
            
            # Model-specific details
            if 'model_specific' in metrics:
                specific = metrics['model_specific']
                self.detailed_text.insert(tk.END, f"• Model Details: {specific}\n")
            
            self.detailed_text.insert(tk.END, f"\n📊 CONFUSION MATRIX:\n")
            self.detailed_text.insert(tk.END, f"{metrics['confusion_matrix']}\n\n")
            
            self.detailed_text.insert(tk.END, f"📈 DETAILED CLASSIFICATION REPORT:\n")
            self.detailed_text.insert(tk.END, f"{metrics['classification_report']}\n\n")

        self.detailed_text.config(state="disabled")

    def update_visualization(self):
        """Update visualization charts"""
        if not self.results:
            return

        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        train_times = [self.results[model]['train_time'] for model in models]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        # Update figure title with split info
        if hasattr(self, 'split_info'):
            self.fig.suptitle(f'Advanced Model Performance Analysis - {self.split_info["ratio"]} Split', 
                             fontsize=16, fontweight='bold')

        # 1. All Metrics Comparison
        x = np.arange(len(models))
        width = 0.18
        
        bars1 = self.ax1.bar(x - 1.5*width, [acc/100 for acc in accuracies], width, 
                            label='Accuracy', color=colors[0], alpha=0.8)
        bars2 = self.ax1.bar(x - 0.5*width, precisions, width, 
                            label='Precision', color=colors[1], alpha=0.8)
        bars3 = self.ax1.bar(x + 0.5*width, recalls, width, 
                            label='Recall', color=colors[2], alpha=0.8)
        bars4 = self.ax1.bar(x + 1.5*width, f1_scores, width, 
                            label='F1-Score', color=colors[3], alpha=0.8)
        
        self.ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Score', fontsize=10)
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models, rotation=45, ha='right')
        self.ax1.legend(fontsize=9)
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                self.ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # Highlight best overall performer
        best_model = self.best_model_info['overall']['model']
        best_idx = models.index(best_model)
        self.ax1.axvline(x=best_idx, color='gold', linestyle='--', alpha=0.7, linewidth=2)

        # 2. Training Time Comparison
        bars = self.ax2.bar(models, train_times, color=colors[:len(models)], alpha=0.8)
        self.ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Training Time (seconds)', fontsize=10)
        self.ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                         f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)
        
        # Highlight fastest
        fastest_model = self.best_model_info['speed']['model']
        fastest_idx = models.index(fastest_model)
        bars[fastest_idx].set_color('lightgreen')
        bars[fastest_idx].set_alpha(1.0)

        # 3. Confusion Matrix for best model
        best_model = self.best_model_info['overall']['model']
        cm = self.results[best_model]['confusion_matrix']
        
        im = self.ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        self.ax3.set_title(f'Confusion Matrix - {best_model}\n(Best Overall Model)', 
                          fontsize=11, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax3.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center", fontsize=10,
                             color="white" if cm[i, j] > thresh else "black")
        
        self.ax3.set_xlabel('Predicted Label', fontsize=10)
        self.ax3.set_ylabel('True Label', fontsize=10)

        # 4. Performance Rankings
        ranks = []
        for model in models:
            avg_rank, _ = self.calculate_model_rank(model)
            ranks.append(avg_rank)
        
        bars = self.ax4.bar(models, ranks, color=colors[:len(models)], alpha=0.8)
        self.ax4.set_title('Overall Model Rankings\n(Lower is Better)', fontsize=12, fontweight='bold')
        self.ax4.set_ylabel('Average Rank', fontsize=10)
        self.ax4.tick_params(axis='x', rotation=45)
        self.ax4.invert_yaxis()  # Lower ranks at top
        
        # Add value labels and highlight best
        for i, (bar, rank) in enumerate(zip(bars, ranks)):
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                         f'#{rank:.1f}', ha='center', va='top', fontsize=9, fontweight='bold')
            
            # Highlight the best ranked model
            if rank == min(ranks):
                bar.set_color('gold')
                bar.set_alpha(1.0)

        # Adjust layout and refresh
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def update_performance_analysis(self):
        """Update training performance analysis with split impact"""
        self.performance_text.config(state="normal")
        self.performance_text.delete("1.0", tk.END)

        if not self.results:
            self.performance_text.insert(tk.END, "No results available.")
            self.performance_text.config(state="disabled")
            return

        analysis = f"""⚡ COMPREHENSIVE TRAINING PERFORMANCE ANALYSIS
{'='*80}

📊 DATASET CONFIGURATION:
Train-Test Split: {self.split_info['ratio']}
Training Samples: {self.split_info['train_size']}
Testing Samples: {self.split_info['test_size']}
Total Dataset Size: {self.split_info['total_size']}

📈 SPLIT IMPACT ANALYSIS:
"""
        split_ratio = self.split_info['ratio']
        train_pct = int(split_ratio.split(':')[0])
        
        if train_pct >= 85:
            analysis += f"• HIGH TRAINING DATA ({train_pct}%): Models have more data to learn patterns\n"
            analysis += f"• Potential for better model complexity and performance\n"
            analysis += f"• Smaller test set may lead to higher variance in evaluation\n"
        elif train_pct >= 75:
            analysis += f"• BALANCED SPLIT ({train_pct}%): Good compromise between training and evaluation\n"
            analysis += f"• Standard practice for most machine learning applications\n"
        else:
            analysis += f"• CONSERVATIVE TRAINING ({train_pct}%): More data reserved for testing\n"
            analysis += f"• More robust evaluation but may limit model learning capability\n"
            analysis += f"• Better for assessing generalization performance\n"
        
        analysis += f"\n📊 TRAINING TIME SUMMARY:\n"

        # Sort by training time
        sorted_by_time = sorted(self.results.items(), key=lambda x: x[1]['train_time'])
        
        analysis += f"{'Model':<15} {'Time (s)':<12} {'Speed Rank':<12} {'Optimization Level':<20}\n"
        analysis += "-" * 70 + "\n"
        
        for i, (model, metrics) in enumerate(sorted_by_time, 1):
            time_val = metrics['train_time']
            if time_val < 0.001:
                speed_desc = "Lightning Fast"
            elif time_val < 0.01:
                speed_desc = "Very Fast"
            elif time_val < 0.1:
                speed_desc = "Fast"
            elif time_val < 1.0:
                speed_desc = "Moderate"
            else:
                speed_desc = "Slow"
                
            analysis += f"{model:<15} {time_val:<12.4f} {i:<12} {speed_desc:<20}\n"

        # Performance vs Accuracy trade-off
        analysis += f"\n🎯 PERFORMANCE vs ACCURACY TRADE-OFF:\n"
        analysis += "=" * 60 + "\n"
        
        for model, metrics in self.results.items():
            accuracy = metrics['accuracy']
            time_val = metrics['train_time']
            efficiency = accuracy / (time_val * 1000)  # Accuracy per millisecond
            
            analysis += f"{model}:\n"
            analysis += f"  • Accuracy: {accuracy:.2f}%\n"
            analysis += f"  • Training Time: {time_val:.4f}s\n"
            analysis += f"  • Efficiency Score: {efficiency:.1f} (acc%/ms)\n\n"

        # Final recommendations
        fastest = min(self.results.items(), key=lambda x: x[1]['train_time'])
        most_accurate = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        
        analysis += f"🚀 RECOMMENDATIONS BASED ON {split_ratio} SPLIT:\n"
        analysis += "=" * 50 + "\n"
        analysis += f"⚡ For speed: {fastest[0]} ({fastest[1]['train_time']:.4f}s, {fastest[1]['accuracy']:.1f}%)\n"
        analysis += f"🎯 For accuracy: {most_accurate[0]} ({most_accurate[1]['accuracy']:.1f}%, {most_accurate[1]['train_time']:.4f}s)\n"
        analysis += f"⚖️ Best overall: {self.best_model_info['overall']['model']} (balanced performance)\n"

        self.performance_text.insert(tk.END, analysis)
        self.performance_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparisonWindow(root)
    root.mainloop()
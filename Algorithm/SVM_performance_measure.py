import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

OUTPUT_FOLDER = "CSV_Files"

class SVMPerformanceWindow:
    def __init__(self, root, callback=None):
        self.root = root
        self.callback = callback  # Add callback for returning results
        self.best_results = None  # Store results
        
        self.root.title("📈 SVM Performance Evaluation")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🔍 Support Vector Machine - Kernel & Hyperparameter Optimization", 
                               font=("Segoe UI", 18, "bold"), bg="#f5f7fa", fg="#333")
        title_label.pack(pady=15)

        # Input Frame
        input_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        
        input_label = tk.Label(input_frame, text="Configuration", font=("Segoe UI", 14, "bold"), 
                               bg="#ffffff", fg="#333")
        input_label.pack(anchor="w", padx=10, pady=5)

        # Kernel selection
        kernel_frame = tk.Frame(input_frame, bg="#ffffff")
        kernel_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(kernel_frame, text="Kernel types to test:", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.kernel_var = tk.StringVar(value="all")
        kernel_options = ["all", "linear", "poly", "rbf", "sigmoid"]
        self.kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.kernel_var, 
                                         values=kernel_options, state="readonly", width=15)
        self.kernel_combo.pack(side=tk.LEFT, padx=10)

        # Top-k features input
        feature_frame = tk.Frame(input_frame, bg="#ffffff")
        feature_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(feature_frame, text="Number of top features (or 'all'):", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.top_k_entry = tk.Entry(feature_frame, font=("Segoe UI", 11), width=10)
        self.top_k_entry.pack(side=tk.LEFT, padx=10)
        self.top_k_entry.insert(0, "all")  # Default value

        # Run Evaluation Button
        self.evaluate_button = ttk.Button(root, text="Run SVM Optimization", 
                                          command=self.evaluate_svm)
        self.evaluate_button.pack(pady=15)

        # Progress bar
        self.progress = ttk.Progressbar(root, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # Status label
        self.status_label = tk.Label(root, text="Ready to run evaluation", 
                                     font=("Segoe UI", 10), bg="#f5f7fa", fg="#666")
        self.status_label.pack(pady=5)

        # Notebook (Tabs)
        notebook = ttk.Notebook(root)
        notebook.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Tab 1: Kernel & Hyperparameter Results
        self.tab1 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab1, text="📊 SVM Results")

        # Table for results
        columns = ("kernel", "C", "gamma", "degree", "accuracy", "train_time")
        self.tree = ttk.Treeview(self.tab1, columns=columns, show="headings", height=12)
        
        self.tree.heading("kernel", text="Kernel")
        self.tree.heading("C", text="C Parameter")
        self.tree.heading("gamma", text="Gamma")
        self.tree.heading("degree", text="Degree")
        self.tree.heading("accuracy", text="Accuracy (%)")
        self.tree.heading("train_time", text="Train Time (s)")
        
        # Column widths
        column_widths = {"kernel": 80, "C": 100, "gamma": 100, "degree": 80, "accuracy": 100, "train_time": 120}
        for col, width in column_widths.items():
            self.tree.column(col, anchor="center", width=width)
        
        self.tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Add scrollbar for table
        tree_scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=self.tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Label for best result
        self.best_label = tk.Label(self.tab1, text="", font=("Segoe UI", 12, "bold"), 
                                   bg="#ffffff", fg="#0a84ff", anchor="w", justify="left")
        self.best_label.pack(pady=10, padx=10, anchor="w")

        # Tab 2: Feature Analysis
        self.tab2 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab2, text="🔍 Feature Analysis")

        self.feature_text = tk.Text(self.tab2, height=15, font=("Courier New", 11), wrap="none")
        self.feature_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.feature_text.config(state="disabled")

        feature_scrollbar = ttk.Scrollbar(self.tab2, command=self.feature_text.yview)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_text.config(yscrollcommand=feature_scrollbar.set)

        # Tab 3: Confusion Matrix & Report
        self.tab3 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab3, text="📑 Detailed Report")

        # Scrollable Text
        self.output_text = tk.Text(self.tab3, height=25, font=("Courier New", 11), wrap="none")
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_text.config(state="disabled")

        # Add scrollbar
        output_scrollbar = ttk.Scrollbar(self.tab3, command=self.output_text.yview)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=output_scrollbar.set)

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()

    def evaluate_svm(self):
        try:
            # Disable button during evaluation
            self.evaluate_button.config(state="disabled")
            
            # Load the filtered dataset
            self.update_status("Loading dataset...")
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            # Get user input for features
            user_input = self.top_k_entry.get().strip().lower()
            kernel_selection = self.kernel_var.get()

            # Determine k features to use
            if user_input == "all" or user_input == "":
                k = None
            else:
                try:
                    k = int(user_input)
                except ValueError:
                    k = None

            # Calculate correlation and select features
            self.update_status("Selecting features...")
            correlations = df.corr(numeric_only=True)["GradeClass"].abs().sort_values(ascending=False)
            correlations = correlations.drop(labels=["GradeClass"], errors='ignore')

            if k is not None and k > 0:
                top_features = correlations.head(k).index.tolist()
            else:
                top_features = correlations.index.tolist()

            top_features.append("GradeClass")
            df_filtered = df[top_features]

            # Split the dataset
            X = df_filtered.drop("GradeClass", axis=1)
            y = df_filtered["GradeClass"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize features (important for SVM)
            self.update_status("Standardizing features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define parameter grids for different kernels
            if kernel_selection == "all":
                kernels_to_test = ['linear', 'poly', 'rbf', 'sigmoid']
            else:
                kernels_to_test = [kernel_selection]

            param_grids = {
                'linear': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'],
                    'degree': [3]  # Not used for linear, but needed for consistency
                },
                'poly': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'degree': [2, 3, 4]
                },
                'rbf': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'degree': [3]  # Not used for RBF
                },
                'sigmoid': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'degree': [3]  # Not used for sigmoid
                }
            }

            best_accuracy = 0
            best_params = {}
            results = []
            total_combinations = sum(len(param_grids[k]['C']) * len(param_grids[k]['gamma']) * 
                                   len(param_grids[k]['degree']) for k in kernels_to_test)
            current_combination = 0

            # Grid search for each kernel
            for kernel in kernels_to_test:
                self.update_status(f"Testing {kernel} kernel...")
                
                for C in param_grids[kernel]['C']:
                    for gamma in param_grids[kernel]['gamma']:
                        for degree in param_grids[kernel]['degree']:
                            current_combination += 1
                            progress = (current_combination / total_combinations) * 100
                            self.progress['value'] = progress
                            self.root.update()
                            
                            start_time = time.time()
                            
                            # Create and train SVM
                            if kernel == 'linear':
                                svm = SVC(kernel=kernel, C=C, random_state=42)
                            elif kernel == 'poly':
                                svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
                            elif kernel == 'rbf':
                                svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                            else:  # sigmoid
                                svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                            
                            svm.fit(X_train_scaled, y_train)
                            train_time = time.time() - start_time
                            
                            y_pred = svm.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred) * 100
                            
                            params = {
                                'kernel': kernel,
                                'C': C,
                                'gamma': gamma if kernel != 'linear' else 'N/A',
                                'degree': degree if kernel == 'poly' else 'N/A',
                                'accuracy': accuracy,
                                'train_time': train_time
                            }
                            results.append(params)
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {
                                    'kernel': kernel,
                                    'C': C,
                                    'gamma': gamma,
                                    'degree': degree
                                }
                                best_svm = svm
                                best_y_pred = y_pred

            # Store results for comparison
            self.best_results = {
                'model': 'SVM',
                'accuracy': best_accuracy,
                'params': best_params,
                'confusion_matrix': confusion_matrix(y_test, best_y_pred),
                'classification_report': classification_report(y_test, best_y_pred, output_dict=True),
                'support_vectors': best_svm.n_support_,
                'num_features': len(top_features) - 1
            }

            # Clear and populate results table
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Sort results by accuracy (descending)
            results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Display top 20 results
            for result in results[:20]:
                self.tree.insert("", "end", values=(
                    result['kernel'],
                    result['C'],
                    result['gamma'],
                    result['degree'],
                    f"{result['accuracy']:.2f}",
                    f"{result['train_time']:.3f}"
                ))

            # Best parameters label
            gamma_display = best_params['gamma'] if best_params['kernel'] != 'linear' else 'N/A'
            degree_display = best_params['degree'] if best_params['kernel'] == 'poly' else 'N/A'
            
            self.best_label.config(
                text=f"✅ Best SVM Configuration:\n"
                     f"🔍 Kernel: {best_params['kernel']}\n"
                     f"📊 C Parameter: {best_params['C']}\n"
                     f"🎛️ Gamma: {gamma_display}\n"
                     f"📈 Degree: {degree_display}\n"
                     f"🎯 Best Accuracy: {best_accuracy:.2f}%\n"
                     f"🔢 Support Vectors: {sum(best_svm.n_support_)}"
            )

            # Feature Analysis
            self.feature_text.config(state="normal")
            self.feature_text.delete("1.0", tk.END)
            
            self.feature_text.insert(tk.END, f"🔍 Feature Analysis (using {len(top_features)-1} features):\n")
            self.feature_text.insert(tk.END, "=" * 60 + "\n\n")
            
            self.feature_text.insert(tk.END, f"📈 Feature Correlation with GradeClass:\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            self.feature_text.insert(tk.END, f"{'Feature':<25} {'Correlation':>15}\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            
            for feature in X.columns:
                if feature in correlations.index:
                    corr_val = correlations[feature]
                    self.feature_text.insert(tk.END, f"{feature:<25} {corr_val:>15.4f}\n")

            self.feature_text.insert(tk.END, f"\n🔢 Dataset Information:\n")
            self.feature_text.insert(tk.END, "-" * 30 + "\n")
            self.feature_text.insert(tk.END, f"Total samples: {len(df_filtered)}\n")
            self.feature_text.insert(tk.END, f"Training samples: {len(X_train)}\n")
            self.feature_text.insert(tk.END, f"Testing samples: {len(X_test)}\n")
            self.feature_text.insert(tk.END, f"Features used: {len(X.columns)}\n")
            
            # Class distribution
            class_dist = y.value_counts().sort_index()
            self.feature_text.insert(tk.END, f"\n📊 Class Distribution:\n")
            self.feature_text.insert(tk.END, "-" * 30 + "\n")
            for class_name, count in class_dist.items():
                percentage = (count / len(y)) * 100
                self.feature_text.insert(tk.END, f"Class {class_name}: {count} ({percentage:.1f}%)\n")
            
            self.feature_text.config(state="disabled")

            # Detailed Report
            cm = confusion_matrix(y_test, best_y_pred)
            report = classification_report(y_test, best_y_pred, digits=2)

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            self.output_text.insert(tk.END, f"🔍 Best SVM Model Results:\n")
            self.output_text.insert(tk.END, "=" * 50 + "\n")
            self.output_text.insert(tk.END, f"Kernel: {best_params['kernel']}\n")
            self.output_text.insert(tk.END, f"C Parameter: {best_params['C']}\n")
            self.output_text.insert(tk.END, f"Gamma: {gamma_display}\n")
            self.output_text.insert(tk.END, f"Degree: {degree_display}\n")
            self.output_text.insert(tk.END, f"Accuracy: {best_accuracy:.2f}%\n")
            self.output_text.insert(tk.END, f"Support Vectors per Class: {list(best_svm.n_support_)}\n")
            self.output_text.insert(tk.END, f"Total Support Vectors: {sum(best_svm.n_support_)}\n\n")

            self.output_text.insert(tk.END, "📊 Confusion Matrix:\n")
            self.output_text.insert(tk.END, f"{cm}\n\n")

            self.output_text.insert(tk.END, "📈 Classification Report (Precision, Recall, F1-Score):\n")
            self.output_text.insert(tk.END, f"{report}\n")

            # Performance comparison across kernels
            self.output_text.insert(tk.END, "⚡ Performance Summary by Kernel:\n")
            self.output_text.insert(tk.END, "-" * 40 + "\n")
            
            kernel_best = {}
            for result in results:
                kernel = result['kernel']
                if kernel not in kernel_best or result['accuracy'] > kernel_best[kernel]['accuracy']:
                    kernel_best[kernel] = result
            
            for kernel, best_result in sorted(kernel_best.items()):
                self.output_text.insert(tk.END, f"{kernel.upper():<10}: {best_result['accuracy']:>6.2f}% "
                                                 f"(C={best_result['C']}, Time: {best_result['train_time']:.3f}s)\n")

            self.output_text.config(state="disabled")

            # Reset progress and status
            self.progress['value'] = 100
            self.update_status("Evaluation completed successfully!")

            # Call callback if provided
            if self.callback:
                self.callback('SVM', self.best_results)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            # Re-enable button
            self.evaluate_button.config(state="normal")
            self.progress['value'] = 0

    def get_best_results(self):
        """Return the results for comparison"""
        return self.best_results


if __name__ == "__main__":
    root = tk.Tk()
    app = SVMPerformanceWindow(root)
    root.mainloop()
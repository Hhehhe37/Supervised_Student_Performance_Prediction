import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

OUTPUT_FOLDER = "CSV_Files"

class DecisionTreePerformanceWindow:
    def __init__(self, root, callback=None):
        self.root = root
        self.callback = callback  # Add callback for returning results
        self.best_results = None  # Store results
        
        self.root.title("📈 Decision Tree Performance Evaluation")
        self.root.geometry("950x750")
        self.root.configure(bg="#f5f7fa")

        # Title
        title_label = tk.Label(root, text="🌳 Decision Tree Classifier - Hyperparameter Optimization", 
                               font=("Segoe UI", 18, "bold"), bg="#f5f7fa", fg="#333")
        title_label.pack(pady=15)

        # Input Frame
        input_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        
        input_label = tk.Label(input_frame, text="Configuration", font=("Segoe UI", 14, "bold"), 
                               bg="#ffffff", fg="#333")
        input_label.pack(anchor="w", padx=10, pady=5)

        # Controls container
        controls_container = tk.Frame(input_frame, bg="#ffffff")
        controls_container.pack(fill=tk.X, padx=10, pady=5)

        # Train-Test Split Ratio Selection
        ratio_frame = tk.Frame(controls_container, bg="#ffffff")
        ratio_frame.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(ratio_frame, text="Train-Test Split Ratio:", 
                 font=("Segoe UI", 11, "bold"), bg="#ffffff").pack(side=tk.LEFT)
        
        self.ratio_var = tk.StringVar()
        self.ratio_combo = ttk.Combobox(ratio_frame, textvariable=self.ratio_var, 
                                       values=["60:40", "70:30", "80:20", "90:10"], 
                                       state="readonly", width=10)
        self.ratio_combo.set("80:20")  # Default value
        self.ratio_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Top-k features input
        feature_frame = tk.Frame(controls_container, bg="#ffffff")
        feature_frame.pack(side=tk.LEFT)
        
        tk.Label(feature_frame, text="Number of top features (or 'all'):", 
                 font=("Segoe UI", 11), bg="#ffffff").pack(side=tk.LEFT)
        
        self.top_k_entry = tk.Entry(feature_frame, font=("Segoe UI", 11), width=10)
        self.top_k_entry.pack(side=tk.LEFT, padx=10)
        self.top_k_entry.insert(0, "all")  # Default value

        # Run Evaluation Button
        self.evaluate_button = ttk.Button(root, text="Run Hyperparameter Optimization", 
                                          command=self.evaluate_decision_tree)
        self.evaluate_button.pack(pady=15)

        # Notebook (Tabs)
        notebook = ttk.Notebook(root)
        notebook.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Tab 1: Hyperparameter Results
        self.tab1 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab1, text="📊 Hyperparameter Results")

        # Configuration info frame
        config_frame = tk.Frame(self.tab1, bg="#ffffff")
        config_frame.pack(pady=10, padx=10, fill="x")

        self.config_label = tk.Label(config_frame, text="", font=("Segoe UI", 10), 
                                     bg="#ffffff", fg="#666", anchor="w", justify="left")
        self.config_label.pack(anchor="w")

        # Table for results
        columns = ("max_depth", "min_samples_split", "min_samples_leaf", "accuracy")
        self.tree = ttk.Treeview(self.tab1, columns=columns, show="headings", height=10)
        
        self.tree.heading("max_depth", text="Max Depth")
        self.tree.heading("min_samples_split", text="Min Samples Split")
        self.tree.heading("min_samples_leaf", text="Min Samples Leaf")
        self.tree.heading("accuracy", text="Accuracy (%)")
        
        for col in columns:
            self.tree.column(col, anchor="center", width=120)
        
        self.tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Add scrollbar for table
        tree_scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=self.tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Label for best result
        self.best_label = tk.Label(self.tab1, text="", font=("Segoe UI", 12, "bold"), 
                                   bg="#ffffff", fg="#0a84ff", anchor="w", justify="left")
        self.best_label.pack(pady=10, padx=10, anchor="w")

        # Tab 2: Feature Importance
        self.tab2 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab2, text="🔍 Feature Importance")

        self.feature_text = tk.Text(self.tab2, height=15, font=("Courier New", 11), wrap="none")
        self.feature_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.feature_text.config(state="disabled")

        # Tab 3: Confusion Matrix & Report
        self.tab3 = tk.Frame(notebook, bg="#ffffff")
        notebook.add(self.tab3, text="📑 Detailed Report")

        # Scrollable Text
        self.output_text = tk.Text(self.tab3, height=25, font=("Courier New", 11), wrap="none")
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output_text.config(state="disabled")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.tab3, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def get_test_size_from_ratio(self, ratio_string):
        """Convert ratio string like '80:20' to test_size float like 0.2"""
        train_ratio, test_ratio = map(int, ratio_string.split(':'))
        return test_ratio / (train_ratio + test_ratio)

    def evaluate_decision_tree(self):
        try:
            # Load the filtered dataset
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            # Get user input for features
            user_input = self.top_k_entry.get().strip().lower()

            # Determine k features to use
            if user_input == "all" or user_input == "":
                k = None
            else:
                try:
                    k = int(user_input)
                except ValueError:
                    k = None

            # Calculate correlation to GradeClass and select features
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

            # Get selected ratio and convert to test_size
            selected_ratio = self.ratio_var.get()
            test_size = self.get_test_size_from_ratio(selected_ratio)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Update configuration display
            features_used = len(top_features) - 1  # Subtract GradeClass
            self.config_label.config(
                text=f"🔧 Configuration: Train-Test Split = {selected_ratio} | Features used: {features_used} | Training samples: {len(X_train)} | Testing samples: {len(X_test)}"
            )

            # Hyperparameter grid
            max_depths = [3, 5, 7, 10, 15, 20, None]
            min_samples_splits = [2, 5, 10, 20]
            min_samples_leafs = [1, 2, 5, 10]

            best_accuracy = 0
            best_params = {}
            results = []

            # Grid search
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    for min_samples_leaf in min_samples_leafs:
                        dt = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42
                        )
                        
                        dt.fit(X_train, y_train)
                        y_pred = dt.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred) * 100
                        
                        params = {
                            'max_depth': max_depth if max_depth else 'None',
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'accuracy': accuracy
                        }
                        results.append(params)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf
                            }
                            best_dt = dt
                            best_y_pred = y_pred

            # Store results for comparison
            self.best_results = {
                'model': 'Decision Tree',
                'accuracy': best_accuracy,
                'params': best_params,
                'confusion_matrix': confusion_matrix(y_test, best_y_pred),
                'classification_report': classification_report(y_test, best_y_pred, output_dict=True),
                'ratio': selected_ratio,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': features_used
            }

            # Clear and populate results table
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Sort results by accuracy (descending)
            results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Display top 20 results
            for result in results[:20]:
                self.tree.insert("", "end", values=(
                    result['max_depth'],
                    result['min_samples_split'],
                    result['min_samples_leaf'],
                    f"{result['accuracy']:.2f}"
                ))

            # Best parameters label
            best_depth = best_params['max_depth'] if best_params['max_depth'] else 'None'
            self.best_label.config(
                text=f"✅ Best Parameters:\n"
                     f"🌳 Max Depth: {best_depth}\n"
                     f"📊 Min Samples Split: {best_params['min_samples_split']}\n"
                     f"🍃 Min Samples Leaf: {best_params['min_samples_leaf']}\n"
                     f"🎯 Best Accuracy: {best_accuracy:.2f}%\n"
                     f"📊 Split Ratio: {selected_ratio}"
            )

            # Feature Importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_dt.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_text.config(state="normal")
            self.feature_text.delete("1.0", tk.END)
            
            self.feature_text.insert(tk.END, f"🔧 Configuration: Train-Test Split = {selected_ratio}\n")
            self.feature_text.insert(tk.END, f"🔍 Feature Importance (using {len(top_features)-1} features):\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            self.feature_text.insert(tk.END, f"{'Feature':<25} {'Importance':>15}\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            
            for _, row in feature_importance.iterrows():
                self.feature_text.insert(tk.END, f"{row['feature']:<25} {row['importance']:>15.4f}\n")
            
            # Show correlation info
            self.feature_text.insert(tk.END, f"\n📈 Feature Correlation with GradeClass:\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            self.feature_text.insert(tk.END, f"{'Feature':<25} {'Correlation':>15}\n")
            self.feature_text.insert(tk.END, "-" * 50 + "\n")
            
            for feature in X.columns:
                if feature in correlations.index:
                    corr_val = correlations[feature]
                    self.feature_text.insert(tk.END, f"{feature:<25} {corr_val:>15.4f}\n")
            
            self.feature_text.config(state="disabled")

            # Confusion Matrix & Classification Report
            cm = confusion_matrix(y_test, best_y_pred)
            report = classification_report(y_test, best_y_pred, digits=2)

            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)

            # Show configuration info
            self.output_text.insert(tk.END, f"🔧 Configuration:\n")
            self.output_text.insert(tk.END, f"   Train-Test Split: {selected_ratio}\n")
            self.output_text.insert(tk.END, f"   Features used: {features_used}\n")
            self.output_text.insert(tk.END, f"   Training samples: {len(X_train)}\n")
            self.output_text.insert(tk.END, f"   Testing samples: {len(X_test)}\n\n")

            self.output_text.insert(tk.END, f"🌳 Best Decision Tree Model Results:\n")
            self.output_text.insert(tk.END, f"Max Depth: {best_depth}\n")
            self.output_text.insert(tk.END, f"Min Samples Split: {best_params['min_samples_split']}\n")
            self.output_text.insert(tk.END, f"Min Samples Leaf: {best_params['min_samples_leaf']}\n")
            self.output_text.insert(tk.END, f"Accuracy: {best_accuracy:.2f}%\n\n")

            self.output_text.insert(tk.END, "📊 Confusion Matrix:\n")
            self.output_text.insert(tk.END, f"{cm}\n\n")

            self.output_text.insert(tk.END, "📈 Classification Report (Precision, Recall, F1-Score):\n")
            self.output_text.insert(tk.END, f"{report}\n")

            # Tree structure info
            self.output_text.insert(tk.END, f"🌳 Tree Structure Information:\n")
            self.output_text.insert(tk.END, f"Number of nodes: {best_dt.tree_.node_count}\n")
            self.output_text.insert(tk.END, f"Number of leaves: {best_dt.tree_.n_leaves}\n")
            self.output_text.insert(tk.END, f"Tree depth: {best_dt.tree_.max_depth}\n")

            self.output_text.config(state="disabled")

            # Call callback if provided
            if self.callback:
                self.callback('Decision Tree', self.best_results)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def get_best_results(self):
        """Return the results for comparison"""
        return self.best_results


if __name__ == "__main__":
    root = tk.Tk()
    app = DecisionTreePerformanceWindow(root)
    root.mainloop()
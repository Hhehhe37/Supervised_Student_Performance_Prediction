import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

OUTPUT_FOLDER = "CSV_Files"

class ModelComparisonWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Model Performance Comparison")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f5f7fa")

        self.results = {}  # Store all model results

        # Title
        title_label = tk.Label(root, text="🏆 Model Performance Comparison", 
                               font=("Segoe UI", 20, "bold"), bg="#f5f7fa", fg="#222")
        title_label.pack(pady=15)

        # Run Comparison Button
        self.compare_button = ttk.Button(root, text="🚀 Run Model Comparison", command=self.run_comparison)
        self.compare_button.pack(pady=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Tab 1: Comparison Table
        self.create_comparison_tab()
        
        # Tab 2: Detailed Metrics
        self.create_detailed_metrics_tab()
        
        # Tab 3: Visualization
        self.create_visualization_tab()

    def create_comparison_tab(self):
        """Create the main comparison table tab"""
        self.tab1 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab1, text="📊 Model Comparison")

        # Comparison Table
        columns = ("Model", "Accuracy (%)", "Precision", "Recall", "F1-Score", "Best Parameter")
        self.comparison_tree = ttk.Treeview(self.tab1, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, anchor="center", width=150)

        self.comparison_tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Winner label
        self.winner_label = tk.Label(self.tab1, text="", font=("Segoe UI", 16, "bold"), 
                                     bg="#ffffff", fg="#228B22")
        self.winner_label.pack(pady=20)

    def create_detailed_metrics_tab(self):
        """Create detailed metrics tab"""
        self.tab2 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab2, text="📑 Detailed Metrics")

        # Scrollable text for detailed results
        self.detailed_text = tk.Text(self.tab2, height=30, font=("Courier New", 10), wrap="word")
        scrollbar2 = ttk.Scrollbar(self.tab2, command=self.detailed_text.yview)
        self.detailed_text.config(yscrollcommand=scrollbar2.set)
        
        self.detailed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        self.detailed_text.config(state="disabled")

    def create_visualization_tab(self):
        """Create visualization tab"""
        self.tab3 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab3, text="📈 Visualization")

        # Create matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.tab3)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run_comparison(self):
        """Run comparison between all models"""
        try:
            # Load data
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            X = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Test different models
            self.results = {}

            # 1. KNN (find best k)
            self.evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # 2. Naive Bayes
            self.evaluate_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # 3. Decision Tree
            self.evaluate_decision_tree(X_train, X_test, y_train, y_test)

            # Update displays
            self.update_comparison_table()
            self.update_detailed_metrics()
            self.update_visualization()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def evaluate_knn(self, X_train, X_test, y_train, y_test):
        """Evaluate KNN with different k values"""
        best_k, best_accuracy = 1, 0
        best_y_pred = None

        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_y_pred = y_pred

        # Calculate metrics for best k
        precision = precision_score(y_test, best_y_pred, average='weighted')
        recall = recall_score(y_test, best_y_pred, average='weighted')
        f1 = f1_score(y_test, best_y_pred, average='weighted')

        self.results['KNN'] = {
            'accuracy': best_accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': f"k={best_k}",
            'y_pred': best_y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, best_y_pred),
            'classification_report': classification_report(y_test, best_y_pred)
        }

    def evaluate_naive_bayes(self, X_train, X_test, y_train, y_test):
        """Evaluate Naive Bayes"""
        nb = GaussianNB()
        nb.fit(X_train, y_train)
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
            'best_param': "Default",
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

    def evaluate_decision_tree(self, X_train, X_test, y_train, y_test):
        """Evaluate Decision Tree"""
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        self.results['Decision Tree'] = {
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_param': "Default",
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

    def update_comparison_table(self):
        """Update the comparison table"""
        # Clear existing data
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)

        best_model = ""
        best_accuracy = 0

        # Insert results
        for model, metrics in self.results.items():
            self.comparison_tree.insert("", "end", values=(
                model,
                f"{metrics['accuracy']:.2f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                metrics['best_param']
            ))
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = model

        # Update winner label
        self.winner_label.config(text=f"🏆 Winner: {best_model} with {best_accuracy:.2f}% accuracy")

    def update_detailed_metrics(self):
        """Update detailed metrics text"""
        self.detailed_text.config(state="normal")
        self.detailed_text.delete("1.0", tk.END)

        for model, metrics in self.results.items():
            self.detailed_text.insert(tk.END, f"{'='*50}\n")
            self.detailed_text.insert(tk.END, f"{model} DETAILED RESULTS\n")
            self.detailed_text.insert(tk.END, f"{'='*50}\n\n")
            
            self.detailed_text.insert(tk.END, f"Accuracy: {metrics['accuracy']:.2f}%\n")
            self.detailed_text.insert(tk.END, f"Best Parameter: {metrics['best_param']}\n\n")
            
            self.detailed_text.insert(tk.END, "Confusion Matrix:\n")
            self.detailed_text.insert(tk.END, f"{metrics['confusion_matrix']}\n\n")
            
            self.detailed_text.insert(tk.END, "Classification Report:\n")
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

        # 1. Accuracy Comparison
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars1 = self.ax1.bar(models, accuracies, color=colors)
        self.ax1.set_title('Model Accuracy Comparison')
        self.ax1.set_ylabel('Accuracy (%)')
        self.ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{acc:.1f}%', ha='center', va='bottom')

        # 2. Multiple Metrics Comparison
        x = np.arange(len(models))
        width = 0.25
        
        self.ax2.bar(x - width, precisions, width, label='Precision', color='#FF6B6B', alpha=0.8)
        self.ax2.bar(x, recalls, width, label='Recall', color='#4ECDC4', alpha=0.8)
        self.ax2.bar(x + width, f1_scores, width, label='F1-Score', color='#45B7D1', alpha=0.8)
        
        self.ax2.set_title('Multiple Metrics Comparison')
        self.ax2.set_ylabel('Score')
        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels(models)
        self.ax2.legend()
        self.ax2.set_ylim(0, 1)

        # 3. Confusion Matrix for best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_model]['confusion_matrix']
        
        im = self.ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        self.ax3.set_title(f'Confusion Matrix - {best_model}')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax3.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
        
        self.ax3.set_xlabel('Predicted Label')
        self.ax3.set_ylabel('True Label')

        # 4. Performance Summary (Radar Chart simulation)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, model in enumerate(models):
            values = [
                self.results[model]['accuracy'] / 100,  # Normalize accuracy
                self.results[model]['precision'],
                self.results[model]['recall'],
                self.results[model]['f1_score']
            ]
            
            self.ax4.plot(metrics_names, values, 'o-', label=model, color=colors[i], linewidth=2, markersize=8)
        
        self.ax4.set_title('Performance Metrics Overview')
        self.ax4.set_ylabel('Score')
        self.ax4.legend()
        self.ax4.set_ylim(0, 1)
        self.ax4.grid(True, alpha=0.3)

        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparisonWindow(root)
    root.mainloop()
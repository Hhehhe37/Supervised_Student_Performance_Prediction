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
        self.root.geometry("1400x900")
        self.root.configure(bg="#f5f7fa")

        self.results = {}  # Store all model results
        self.best_model_info = {}  # Store best model for each metric

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
        
        # Tab 2: Best Model Analysis
        self.create_best_model_tab()
        
        # Tab 3: Detailed Metrics
        self.create_detailed_metrics_tab()
        
        # Tab 4: Visualization
        self.create_visualization_tab()

    def create_comparison_tab(self):
        """Create the main comparison table tab"""
        self.tab1 = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(self.tab1, text="📊 Model Comparison")

        # Comparison Table
        columns = ("Model", "Accuracy (%)", "Precision", "Recall", "F1-Score", "Best Parameter", "Overall Rank")
        self.comparison_tree = ttk.Treeview(self.tab1, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.comparison_tree.heading(col, text=col)
            if col == "Model":
                self.comparison_tree.column(col, anchor="center", width=120)
            elif col == "Best Parameter":
                self.comparison_tree.column(col, anchor="center", width=120)
            else:
                self.comparison_tree.column(col, anchor="center", width=100)

        self.comparison_tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Winner labels frame
        self.winners_frame = tk.Frame(self.tab1, bg="#ffffff")
        self.winners_frame.pack(pady=20, fill=tk.X)
        
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
        self.best_model_text = tk.Text(self.tab2, height=15, font=("Segoe UI", 11), wrap="word")
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
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.tab4)
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

            # Find best models for each metric
            self.find_best_models()

            # Update displays
            self.update_comparison_table()
            self.update_best_model_analysis()
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
        
        # Calculate overall best (weighted average of all metrics)
        for model, metrics in self.results.items():
            # Normalize accuracy to 0-1 scale for fair comparison
            normalized_accuracy = metrics['accuracy'] / 100
            
            # Calculate weighted average (you can adjust weights based on importance)
            overall_score = (normalized_accuracy * 0.3 + 
                           metrics['precision'] * 0.25 + 
                           metrics['recall'] * 0.25 + 
                           metrics['f1_score'] * 0.2)
            
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
        
        # Calculate average rank (lower is better)
        ranks = []
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            metric_key = metric if metric != 'accuracy' else 'accuracy'
            sorted_models = sorted(self.results.keys(), 
                                 key=lambda x: self.results[x][metric_key] if metric_key != 'accuracy' 
                                 else self.results[x]['accuracy'], reverse=True)
            ranks.append(sorted_models.index(model) + 1)
        
        avg_rank = sum(ranks) / len(ranks)
        return avg_rank, best_count

    def update_comparison_table(self):
        """Update the comparison table"""
        # Clear existing data
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)

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
                f"#{int(avg_rank)} ({best_count} wins)"
            ))

        # Update winner labels
        overall_winner = self.best_model_info['overall']['model']
        overall_score = self.best_model_info['overall']['score']
        self.overall_winner_label.config(
            text=f"🏆 Overall Best Model: {overall_winner} (Score: {overall_score:.3f})"
        )

        # Individual metric winners
        winners_text = (f"📊 Best by Accuracy: {self.best_model_info['accuracy']['model']} "
                       f"({self.best_model_info['accuracy']['score']:.2f}%)\n"
                       f"🎯 Best by Precision: {self.best_model_info['precision']['model']} "
                       f"({self.best_model_info['precision']['score']:.3f})\n"
                       f"📈 Best by Recall: {self.best_model_info['recall']['model']} "
                       f"({self.best_model_info['recall']['score']:.3f})\n"
                       f"⚖️ Best by F1-Score: {self.best_model_info['f1_score']['model']} "
                       f"({self.best_model_info['f1_score']['score']:.3f})")
        self.metric_winners_label.config(text=winners_text)

    def update_best_model_analysis(self):
        """Update best model analysis"""
        self.best_model_text.config(state="normal")
        self.best_model_text.delete("1.0", tk.END)

        best_model = self.best_model_info['overall']['model']
        best_metrics = self.results[best_model]

        analysis = f"""🏆 BEST OVERALL MODEL ANALYSIS
{'='*60}

Model: {best_model}
Overall Score: {self.best_model_info['overall']['score']:.3f}
Best Parameter: {best_metrics['best_param']}

PERFORMANCE METRICS:
• Accuracy: {best_metrics['accuracy']:.2f}%
• Precision: {best_metrics['precision']:.3f}
• Recall: {best_metrics['recall']:.3f}
• F1-Score: {best_metrics['f1_score']:.3f}

WHY THIS MODEL IS BEST:
"""

        # Analyze why this model is best
        wins = []
        if self.best_model_info['accuracy']['model'] == best_model:
            wins.append("✓ Highest Accuracy")
        if self.best_model_info['precision']['model'] == best_model:
            wins.append("✓ Highest Precision")
        if self.best_model_info['recall']['model'] == best_model:
            wins.append("✓ Highest Recall")
        if self.best_model_info['f1_score']['model'] == best_model:
            wins.append("✓ Highest F1-Score")

        if wins:
            analysis += "\n".join(wins) + "\n\n"
        else:
            analysis += "• Best balanced performance across all metrics\n\n"

        analysis += f"""DETAILED CLASSIFICATION REPORT:
{best_metrics['classification_report']}

CONFUSION MATRIX:
{best_metrics['confusion_matrix']}

RECOMMENDATION:
Use {best_model} with parameter {best_metrics['best_param']} for your student performance prediction task.
This model provides the best balance of accuracy, precision, recall, and F1-score.
"""

        self.best_model_text.insert(tk.END, analysis)
        self.best_model_text.config(state="disabled")

    def update_detailed_metrics(self):
        """Update detailed metrics text"""
        self.detailed_text.config(state="normal")
        self.detailed_text.delete("1.0", tk.END)

        # Sort models by overall performance
        sorted_models = sorted(self.results.keys(), 
                             key=lambda x: self.best_model_info['overall']['score'] 
                             if self.best_model_info['overall']['model'] == x else 0, 
                             reverse=True)

        for i, model in enumerate(sorted_models, 1):
            metrics = self.results[model]
            self.detailed_text.insert(tk.END, f"{'='*60}\n")
            self.detailed_text.insert(tk.END, f"#{i} - {model} DETAILED RESULTS\n")
            self.detailed_text.insert(tk.END, f"{'='*60}\n\n")
            
            self.detailed_text.insert(tk.END, f"Accuracy: {metrics['accuracy']:.2f}%\n")
            self.detailed_text.insert(tk.END, f"Precision: {metrics['precision']:.3f}\n")
            self.detailed_text.insert(tk.END, f"Recall: {metrics['recall']:.3f}\n")
            self.detailed_text.insert(tk.END, f"F1-Score: {metrics['f1_score']:.3f}\n")
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

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        # 1. All Metrics Comparison
        x = np.arange(len(models))
        width = 0.2
        
        self.ax1.bar(x - 1.5*width, [acc/100 for acc in accuracies], width, label='Accuracy', color=colors[0], alpha=0.8)
        self.ax1.bar(x - 0.5*width, precisions, width, label='Precision', color=colors[1], alpha=0.8)
        self.ax1.bar(x + 0.5*width, recalls, width, label='Recall', color=colors[2], alpha=0.8)
        self.ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color=colors[3], alpha=0.8)
        
        self.ax1.set_title('All Metrics Comparison')
        self.ax1.set_ylabel('Score')
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models)
        self.ax1.legend()
        self.ax1.set_ylim(0, 1)
        
        # Highlight best performers
        best_model = self.best_model_info['overall']['model']
        best_idx = models.index(best_model)
        self.ax1.axvline(x=best_idx, color='gold', linestyle='--', alpha=0.7, linewidth=2)

        # 2. Model Rankings
        ranks = []
        for model in models:
            avg_rank, _ = self.calculate_model_rank(model)
            ranks.append(avg_rank)
        
        bars = self.ax2.bar(models, ranks, color=colors[:len(models)])
        self.ax2.set_title('Model Rankings (Lower is Better)')
        self.ax2.set_ylabel('Average Rank')
        self.ax2.invert_yaxis()  # Lower ranks at top
        
        # Add value labels
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{rank:.1f}', ha='center', va='bottom')

        # 3. Confusion Matrix for best model
        best_model = self.best_model_info['overall']['model']
        cm = self.results[best_model]['confusion_matrix']
        
        im = self.ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        self.ax3.set_title(f'Confusion Matrix - {best_model} (Best Overall)')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax3.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
        
        self.ax3.set_xlabel('Predicted Label')
        self.ax3.set_ylabel('True Label')

        # 4. Metric Winners Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        winner_counts = {}
        
        for model in models:
            winner_counts[model] = 0
            if self.best_model_info['accuracy']['model'] == model:
                winner_counts[model] += 1
            if self.best_model_info['precision']['model'] == model:
                winner_counts[model] += 1
            if self.best_model_info['recall']['model'] == model:
                winner_counts[model] += 1
            if self.best_model_info['f1_score']['model'] == model:
                winner_counts[model] += 1
        
        bars = self.ax4.bar(models, list(winner_counts.values()), color=colors[:len(models)])
        self.ax4.set_title('Number of Metrics Won by Each Model')
        self.ax4.set_ylabel('Number of Best Scores')
        self.ax4.set_ylim(0, 4)
        
        # Add value labels
        for bar, count in zip(bars, winner_counts.values()):
            if count > 0:
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                             f'{count}', ha='center', va='bottom', fontweight='bold')

        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparisonWindow(root)
    root.mainloop()
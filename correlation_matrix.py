import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationMatrixView:

    def __init__(self, file_path="Corrected_StudentsPerformance.csv"):
        self.file_path = file_path

    def show_correlation(self):
        df = pd.read_csv(self.file_path)  

        correlation_matrix = df.corr(numeric_only=True)  # Only include numeric columns

        # Display correlation matrix
        print(correlation_matrix)

        # Visualize it with a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
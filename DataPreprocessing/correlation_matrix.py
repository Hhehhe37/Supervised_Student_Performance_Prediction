import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "CSV_Files"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class CorrelationMatrixView:

    def __init__(self, file_path=os.path.join(OUTPUT_FOLDER, "Corrected_StudentsPerformance.csv")):
        self.file_path = file_path

    def show_correlation(self):
        df = pd.read_csv(self.file_path)  

        correlation_matrix = df.corr(numeric_only=True)  # Only include numeric columns

        # Display correlation matrix
        print(correlation_matrix)

        #Filter Correlations >= 0.1 or =< 0.1, excluding GradeClass Itself
        if 'GradeClass' in correlation_matrix:
            grade_corr = correlation_matrix['GradeClass']
            selectedFeatures = grade_corr[(grade_corr > 0.1) | (grade_corr < -0.1)].drop(['GradeClass', 'GPA'], errors='ignore').index.tolist() # Store the attribute to a list

            selectedFeatures.append('GradeClass')

            filtered_df = df[selectedFeatures]
            filtered_df.to_csv(os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv"), index=False)


        # Visualize it with a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
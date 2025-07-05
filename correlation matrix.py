import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your CSV file
df = pd.read_csv("Student_performance_data _.csv")  # replace with your file name

# Step 2: Create correlation matrix
correlation_matrix = df.corr(numeric_only=True)  # Only include numeric columns

# Step 3: Display correlation matrix
print(correlation_matrix)

# Step 4: Visualize it with a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
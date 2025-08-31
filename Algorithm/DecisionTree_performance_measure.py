import tkinter as tk
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

OUTPUT_FOLDER = "CSV_Files"

class DecisionTreePerformanceWindow:
    def __init__(self, root):

        #entry field of top features
        self.top_k_label = tk.Label(root, text="Enter the number of top features to consider: ", font=("Arial", 12))
        self.top_k_label.pack(pady=10)

        self.top_k_entry = tk.Entry(root, font=("Arial", 12))
        self.top_k_entry.pack(pady=5)

        #create a TKinter GUI window for Decision Tree performance evaluation
        self.root = root
        self.root.title("Decision Tree Performance Measure") #set window title
        self.root.geometry("1280x800") #set window size

        #create and display a window
        self.label = tk.Label(root, text="Decision Tree Performance Measure", font=("Arial", 12))
        self.label.pack(pady=10)

        #create and display a button to trigger model evaluation
        self.evaluate_button = tk.Button(root, text="Evaluate Decision Tree", command=self.evaluate_decision_tree)
        self.evaluate_button.pack(pady=10)

        #create and display a text widget to show output
        self.output_text = tk.Text(root, height=20, width=50)
        self.output_text.pack(pady=10)
        self.output_text.config(state=tk.DISABLED) #disable text widget to prevent user input

    def evaluate_decision_tree(self):
        try:
            #load the filtered dataset that was preprocessed
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)

            #read user input
            user_input = self.top_k_entry.get().strip().lower()

            # determine k features to use
            if user_input == "all" or user_input == "":
                k = None #if user input is "all" or empty, use all features
            else:
                try:
                    k = int(user_input) #if user input is a valid integer, use that number of features
                except ValueError:
                    k = None #if invalid, fallback to using all top features for the classification

            #calculate correlation to GradeClass
            correlations = df.corr(numeric_only = True)["GradeClass"].abs().sort_values(ascending=False)
            correlations = correlations.drop(labels=["GradeClass"], errors='ignore')

            #select top k features
            if k is not None and k > 0:
                top_features = correlations.head(k).index.tolist()
            else:
                top_features = correlations.index.tolist()

            #add target column
            top_features.append("GradeClass")

            #filter the dataframe
            df = df[top_features]

            #split the dataset into features and target variable
            X = df.drop("GradeClass", axis=1)
            y = df["GradeClass"]

            #split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #initialize the Decision Tree model
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)

            #make predictions on the testing set
            y_pred = model.predict(X_test)

            #calculate the performance metrics
            accuracy = accuracy_score(y_test, y_pred) * 100

            #display the result in the text box
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Accuracy: {accuracy:.2f}%")
            self.output_text.config(state="disabled")

        #handle the errors
        except Exception as e:
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Error: {str(e)}")
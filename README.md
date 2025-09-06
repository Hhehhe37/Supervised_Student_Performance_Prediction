# Supervised_Student_Performance_Prediction

1. Need to install seaborn (pip install seaborn)
2. Install pandas (python -m pip install pandas)

What included in the ui:
PredicitonModel.py
1. This file contains the model that is used to predict the student's performance based on the given features.
2. The model is trained using the data from the dataset and is used to make predictions on new data.
3. First input require users to enter the number of absences by certain student.
4. Second input require users to enter the number of study time weekly by certain student.
5. Third input require users to enter whether the student has tutoring or not.
6. Fourth input require users to enter the level of parental support of the student.
7. Eventually click the predict button to obtain the final prediction result.

PreprocessingUI.py
1. This file contains the preprocessing steps that are used to clean and prepare the data for the model.
2. The preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical variables.
3. The file can allows users to clearly understand what preprocessing steps that we have done in this model.
4. The file can generate the accuracy metrics result of each algorithm with selecting different ratio of train and test data.
5. The final prediction model that being deployed is chosen by the accuracy metrics result of the algorithm in term of accuracy and time consumed.
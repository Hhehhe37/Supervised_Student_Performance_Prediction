import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

OUTPUT_FOLDER = "CSV_Files"

class SVMGradePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Student Performance Prediction System")
        self.root.geometry("800x700")
        self.root.configure(bg="#f8f9fa")
        
        # Initialize model variables
        self.model = None
        self.scaler = None
        self.model_trained = False
        
        # Phase tracking
        self.current_phase = "initiation"  # initiation, data_input, prediction
        self.input_data = {}
        self.input_step = 0  # Track which input we're collecting
        
        # Load or train model at startup
        self.load_or_train_model()
        
        # Start with initiation phase
        self.create_initiation_phase()
        
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            # Try to load existing model
            model_files = [f for f in os.listdir("models") if f.startswith("svm_grade_model_")]
            if model_files:
                latest_model = sorted(model_files)[-1]
                scaler_file = latest_model.replace("svm_grade_model_", "svm_scaler_")
                
                self.model = joblib.load(f"models/{latest_model}")
                self.scaler = joblib.load(f"models/{scaler_file}")
                self.model_trained = True
                print("Model loaded successfully!")
            else:
                self.train_background_model()
        except:
            self.train_background_model()
    
    def train_background_model(self):
        """Train model in background"""
        try:
            # Create sample dataset if needed
            self.create_sample_dataset()
            
            # Load dataset
            file_path = os.path.join(OUTPUT_FOLDER, "FilteredStudentPerformance.csv")
            df = pd.read_csv(file_path)
            df = df.dropna()
            
            # Prepare data
            X = df[['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport']]
            y = df['GradeClass']
            
            # Split data 90:10
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train SVM with optimized parameters
            self.model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
            self.model.fit(X_train_scaled, y_train)
            self.model_trained = True
            
            # Save model
            self.save_model()
            print("Model trained and saved successfully!")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    def create_sample_dataset(self):
        """Create sample dataset if it doesn't exist"""
        if not os.path.exists("CSV_Files"):
            os.makedirs("CSV_Files")
        
        file_path = os.path.join("CSV_Files", "FilteredStudentPerformance.csv")
        
        if not os.path.exists(file_path):
            np.random.seed(42)
            n_samples = 1000
            
            study_time = np.random.normal(15, 5, n_samples)
            study_time = np.clip(study_time, 0, 30)
            
            absences = np.random.poisson(8, n_samples)
            absences = np.clip(absences, 0, 30)
            
            tutoring = np.random.binomial(1, 0.3, n_samples)
            parental_support = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
            
            # Generate grade class based on logical rules
            grade_class = np.zeros(n_samples)
            
            for i in range(n_samples):
                score = 0
                if study_time[i] >= 20: score += 4
                elif study_time[i] >= 15: score += 3
                elif study_time[i] >= 10: score += 2
                elif study_time[i] >= 5: score += 1
                
                if absences[i] <= 3: score += 2
                elif absences[i] <= 7: score += 1
                elif absences[i] <= 15: score += 0
                elif absences[i] <= 25: score -= 1
                else: score -= 2
                
                if tutoring[i] == 1: score += 2
                score += parental_support[i] - 1
                score += np.random.normal(0, 1)
                
                if score <= 2: grade_class[i] = 4   # F (worst)
                elif score <= 4: grade_class[i] = 3 # D
                elif score <= 6: grade_class[i] = 2 # C
                elif score <= 8: grade_class[i] = 1 # B
                else: grade_class[i] = 0            # A (best)
            
            df = pd.DataFrame({
                'StudyTimeWeekly': study_time,
                'Absences': absences.astype(int),
                'Tutoring': tutoring,
                'ParentalSupport': parental_support,
                'GradeClass': grade_class.astype(int)
            })
            
            df.to_csv(file_path, index=False)
    
    def save_model(self):
        """Save the trained model"""
        try:
            if not os.path.exists("models"):
                os.makedirs("models")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(self.model, f"models/svm_grade_model_{timestamp}.pkl")
            joblib.dump(self.scaler, f"models/svm_scaler_{timestamp}.pkl")
        except Exception as e:
            print(f"Warning: Could not save model: {str(e)}")
    
    def clear_screen(self):
        """Clear all widgets from the screen"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_initiation_phase(self):
        """Phase 1: Initiation - Welcome and decision to proceed"""
        self.clear_screen()
        self.current_phase = "initiation"
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f8f9fa", padx=40, pady=40)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🎯 Student Performance Prediction System", 
                              font=("Segoe UI", 24, "bold"), 
                              bg="#f8f9fa", 
                              fg="#2c3e50")
        title_label.pack(pady=30)
        
        # Welcome message
        welcome_text = """Welcome to the Student Performance Prediction System!

This application uses machine learning to predict a student's academic performance 
based on four key factors:

• Number of absences
• Weekly study hours  
• Tutoring status
• Parental support level

The system will guide you through a simple 3-step process to make an accurate prediction."""
        
        welcome_label = tk.Label(main_frame, 
                                text=welcome_text,
                                font=("Segoe UI", 12),
                                bg="#f8f9fa",
                                fg="#34495e",
                                justify="left",
                                wraplength=600)
        welcome_label.pack(pady=30)
        
        # Decision frame
        decision_frame = tk.Frame(main_frame, bg="#f8f9fa")
        decision_frame.pack(pady=40)
        
        question_label = tk.Label(decision_frame,
                                 text="Would you like to proceed with a prediction?",
                                 font=("Segoe UI", 14, "bold"),
                                 bg="#f8f9fa",
                                 fg="#2c3e50")
        question_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(decision_frame, bg="#f8f9fa")
        button_frame.pack(pady=20)
        
        proceed_btn = tk.Button(button_frame,
                               text="✅ Yes, Let's Start!",
                               font=("Segoe UI", 12, "bold"),
                               bg="#27ae60",
                               fg="white",
                               padx=20,
                               pady=10,
                               command=self.proceed_to_data_input,
                               cursor="hand2")
        proceed_btn.pack(side=tk.LEFT, padx=10)
        
        exit_btn = tk.Button(button_frame,
                            text="❌ No, Exit",
                            font=("Segoe UI", 12, "bold"),
                            bg="#e74c3c",
                            fg="white",
                            padx=20,
                            pady=10,
                            command=self.exit_application,
                            cursor="hand2")
        exit_btn.pack(side=tk.LEFT, padx=10)
        
        # Status
        if not self.model_trained:
            status_label = tk.Label(main_frame,
                                   text="⚠️ Model is still loading in background...",
                                   font=("Segoe UI", 10),
                                   bg="#f8f9fa",
                                   fg="#f39c12")
            status_label.pack(pady=10)
    
    def proceed_to_data_input(self):
        """Move to data input phase"""
        if not self.model_trained:
            messagebox.showwarning("Please Wait", "Model is still loading. Please wait a moment and try again.")
            return
        
        self.input_data = {}
        self.input_step = 0
        self.create_data_input_phase()
    
    def exit_application(self):
        """Exit the application gracefully"""
        self.clear_screen()
        
        # Thank you message
        thank_you_frame = tk.Frame(self.root, bg="#f8f9fa", padx=40, pady=40)
        thank_you_frame.pack(fill=tk.BOTH, expand=True)
        
        thank_you_label = tk.Label(thank_you_frame,
                                  text="Thank you for using the Student Performance Prediction System!",
                                  font=("Segoe UI", 18, "bold"),
                                  bg="#f8f9fa",
                                  fg="#2c3e50")
        thank_you_label.pack(pady=50)
        
        goodbye_label = tk.Label(thank_you_frame,
                                text="Have a great day! 🎓",
                                font=("Segoe UI", 14),
                                bg="#f8f9fa",
                                fg="#7f8c8d")
        goodbye_label.pack(pady=20)
        
        # Auto-close after 2 seconds
        self.root.after(2000, self.root.destroy)
    
    def create_data_input_phase(self):
        """Phase 2: Data Input and Validation"""
        self.clear_screen()
        self.current_phase = "data_input"
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f8f9fa", padx=40, pady=40)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame,
                              text="📝 Data Input Phase",
                              font=("Segoe UI", 20, "bold"),
                              bg="#f8f9fa",
                              fg="#2c3e50")
        title_label.pack(pady=20)
        
        # Progress indicator
        progress_text = f"Step {self.input_step + 1} of 4"
        progress_label = tk.Label(main_frame,
                                 text=progress_text,
                                 font=("Segoe UI", 12),
                                 bg="#f8f9fa",
                                 fg="#7f8c8d")
        progress_label.pack(pady=5)
        
        # Progress bar
        progress_frame = tk.Frame(main_frame, bg="#f8f9fa")
        progress_frame.pack(pady=10)
        
        for i in range(4):
            color = "#3498db" if i <= self.input_step else "#ecf0f1"
            step_bar = tk.Frame(progress_frame, bg=color, width=80, height=8)
            step_bar.pack(side=tk.LEFT, padx=2)
        
        # Input section
        input_frame = tk.LabelFrame(main_frame,
                                   text="Please provide the following information:",
                                   font=("Segoe UI", 12, "bold"),
                                   bg="#ffffff",
                                   fg="#2c3e50",
                                   padx=20,
                                   pady=20)
        input_frame.pack(pady=30, fill=tk.X)
        
        # Current input field
        self.create_current_input_field(input_frame)
    
    def create_current_input_field(self, parent):
        """Create the current input field based on input_step"""
        
        input_configs = [
            {
                "label": "❌ Number of Absences",
                "description": "How many days has the student been absent this month?",
                "hint": "Enter a number between 0 and 30",
                "key": "absences",
                "validation": self.validate_absences
            },
            {
                "label": "📚 Weekly Study Hours",
                "description": "How many hours per week does the student spend studying?",
                "hint": "Enter a number between 0 and 30 (can include decimals)",
                "key": "study_time",
                "validation": self.validate_study_time
            },
            {
                "label": "👨‍🏫 Tutoring Status",
                "description": "Does the student receive tutoring support?",
                "hint": "Enter 0 for No or 1 for Yes",
                "key": "tutoring",
                "validation": self.validate_tutoring
            },
            {
                "label": "👪 Parental Support Level",
                "description": "What is the level of parental support?",
                "hint": "Enter 1 for Low, 2 for Medium, or 3 for High",
                "key": "parental_support",
                "validation": self.validate_parental_support
            }
        ]
        
        config = input_configs[self.input_step]
        
        # Question label
        question_label = tk.Label(parent,
                                 text=config["label"],
                                 font=("Segoe UI", 14, "bold"),
                                 bg="#ffffff",
                                 fg="#2c3e50")
        question_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(parent,
                             text=config["description"],
                             font=("Segoe UI", 11),
                             bg="#ffffff",
                             fg="#34495e")
        desc_label.pack(pady=5)
        
        # Input field
        input_frame = tk.Frame(parent, bg="#ffffff")
        input_frame.pack(pady=15)
        
        self.current_input_var = tk.StringVar()
        self.current_input_entry = tk.Entry(input_frame,
                                           textvariable=self.current_input_var,
                                           font=("Segoe UI", 12),
                                           width=20,
                                           justify="center")
        self.current_input_entry.pack(pady=5)
        self.current_input_entry.focus()
        
        # Hint
        hint_label = tk.Label(input_frame,
                             text=config["hint"],
                             font=("Segoe UI", 9),
                             bg="#ffffff",
                             fg="#7f8c8d")
        hint_label.pack(pady=5)
        
        # Validation message
        self.validation_label = tk.Label(parent,
                                        text="",
                                        font=("Segoe UI", 10),
                                        bg="#ffffff")
        self.validation_label.pack(pady=10)
        
        # Next button
        button_frame = tk.Frame(parent, bg="#ffffff")
        button_frame.pack(pady=20)
        
        next_btn = tk.Button(button_frame,
                            text="➡️ Next" if self.input_step < 3 else "🎯 Make Prediction",
                            font=("Segoe UI", 12, "bold"),
                            bg="#3498db",
                            fg="white",
                            padx=20,
                            pady=8,
                            command=lambda: self.process_current_input(config),
                            cursor="hand2")
        next_btn.pack(side=tk.LEFT, padx=5)
        
        # Back button (except for first step)
        if self.input_step > 0:
            back_btn = tk.Button(button_frame,
                                text="⬅️ Back",
                                font=("Segoe UI", 12, "bold"),
                                bg="#95a5a6",
                                fg="white",
                                padx=20,
                                pady=8,
                                command=self.go_back,
                                cursor="hand2")
            back_btn.pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key
        self.current_input_entry.bind('<Return>', lambda e: self.process_current_input(config))
    
    def process_current_input(self, config):
        """Process the current input and move to next step or prediction"""
        value = self.current_input_var.get().strip()
        
        # Validate input
        is_valid, error_msg, processed_value = config["validation"](value)
        
        if not is_valid:
            self.validation_label.config(text=f"❌ {error_msg}", fg="#e74c3c")
            return
        
        # Store valid input
        self.input_data[config["key"]] = processed_value
        self.validation_label.config(text="✅ Valid input!", fg="#27ae60")
        
        # Move to next step or prediction
        if self.input_step < 3:
            self.input_step += 1
            self.create_data_input_phase()
        else:
            self.create_prediction_phase()
    
    def go_back(self):
        """Go back to previous input step"""
        if self.input_step > 0:
            self.input_step -= 1
            self.create_data_input_phase()
    
    def validate_absences(self, value):
        """Validate absences input"""
        try:
            absences = int(value)
            if absences < 0 or absences > 30:
                return False, "Absences must be between 0 and 30", None
            return True, "", absences
        except ValueError:
            return False, "Please enter a valid whole number", None
    
    def validate_study_time(self, value):
        """Validate study time input"""
        try:
            study_time = float(value)
            if study_time < 0 or study_time > 30:
                return False, "Study time must be between 0 and 30 hours", None
            return True, "", study_time
        except ValueError:
            return False, "Please enter a valid number", None
    
    def validate_tutoring(self, value):
        """Validate tutoring input"""
        try:
            tutoring = int(value)
            if tutoring not in [0, 1]:
                return False, "Please enter 0 for No or 1 for Yes", None
            return True, "", tutoring
        except ValueError:
            return False, "Please enter 0 or 1", None
    
    def validate_parental_support(self, value):
        """Validate parental support input"""
        try:
            support = int(value)
            if support not in [1, 2, 3]:
                return False, "Please enter 1 (Low), 2 (Medium), or 3 (High)", None
            return True, "", support
        except ValueError:
            return False, "Please enter 1, 2, or 3", None
    
    def create_prediction_phase(self):
        """Phase 3: Prediction and Conclusion"""
        self.clear_screen()
        self.current_phase = "prediction"
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f8f9fa", padx=40, pady=40)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame,
                              text="🎯 Prediction Results",
                              font=("Segoe UI", 20, "bold"),
                              bg="#f8f9fa",
                              fg="#2c3e50")
        title_label.pack(pady=20)
        
        # Make prediction
        try:
            # Prepare input data
            input_array = np.array([[
                self.input_data["study_time"],
                self.input_data["absences"],
                self.input_data["tutoring"],
                self.input_data["parental_support"]
            ]])
            
            # Scale and predict
            input_scaled = self.scaler.transform(input_array)
            prediction = int(self.model.predict(input_scaled)[0])
            prediction_proba = self.model.predict_proba(input_scaled)[0]
            
            # Display input summary
            summary_frame = tk.LabelFrame(main_frame,
                                         text="📋 Input Summary",
                                         font=("Segoe UI", 12, "bold"),
                                         bg="#ffffff",
                                         fg="#2c3e50",
                                         padx=20,
                                         pady=15)
            summary_frame.pack(pady=20, fill=tk.X)
            
            summary_text = f"""• Absences: {self.input_data["absences"]} days
• Study Time: {self.input_data["study_time"]} hours/week
• Tutoring: {'Yes' if self.input_data["tutoring"] else 'No'}
• Parental Support: {['', 'Low', 'Medium', 'High'][self.input_data["parental_support"]]}"""
            
            summary_label = tk.Label(summary_frame,
                                    text=summary_text,
                                    font=("Segoe UI", 11),
                                    bg="#ffffff",
                                    fg="#34495e",
                                    justify="left")
            summary_label.pack()
            
            # Display prediction
            result_frame = tk.LabelFrame(main_frame,
                                        text="🎓 Prediction Result",
                                        font=("Segoe UI", 12, "bold"),
                                        bg="#ffffff",
                                        fg="#2c3e50",
                                        padx=20,
                                        pady=20)
            result_frame.pack(pady=20, fill=tk.X)
            
            grade_descriptions = {
                0: ("Excellent Performance", "A grade", "#27ae60"),
                1: ("Good Performance", "B grade", "#2ecc71"),
                2: ("Average Performance", "C grade", "#f1c40f"),
                3: ("Below Average", "D grade", "#f39c12"),
                4: ("Poor Performance", "F grade", "#e74c3c")
            }

            
            desc, grade_range, color = grade_descriptions[prediction]
            
            # Prediction display
            pred_label = tk.Label(result_frame,
                                 text=f"Grade Class: {prediction}",
                                 font=("Segoe UI", 18, "bold"),
                                 bg="#ffffff",
                                 fg=color)
            pred_label.pack(pady=10)
            
            desc_label = tk.Label(result_frame,
                                 text=f"{desc} ({grade_range})",
                                 font=("Segoe UI", 14),
                                 bg="#ffffff",
                                 fg="#2c3e50")
            desc_label.pack(pady=5)
            
        except Exception as e:
            error_label = tk.Label(main_frame,
                                  text=f"❌ Error making prediction: {str(e)}",
                                  font=("Segoe UI", 12),
                                  bg="#f8f9fa",
                                  fg="#e74c3c")
            error_label.pack(pady=20)
        
        # Options
        option_frame = tk.Frame(main_frame, bg="#f8f9fa")
        option_frame.pack(pady=40)
        
        option_label = tk.Label(option_frame,
                               text="Would you like to make another prediction?",
                               font=("Segoe UI", 14, "bold"),
                               bg="#f8f9fa",
                               fg="#2c3e50")
        option_label.pack(pady=20)
        
        button_frame = tk.Frame(option_frame, bg="#f8f9fa")
        button_frame.pack(pady=20)
        
        another_btn = tk.Button(button_frame,
                               text="🔄 Yes, Another Prediction",
                               font=("Segoe UI", 12, "bold"),
                               bg="#3498db",
                               fg="white",
                               padx=20,
                               pady=10,
                               command=self.proceed_to_data_input,
                               cursor="hand2")
        another_btn.pack(side=tk.LEFT, padx=10)
        
        finish_btn = tk.Button(button_frame,
                              text="✅ No, I'm Finished",
                              font=("Segoe UI", 12, "bold"),
                              bg="#27ae60",
                              fg="white",
                              padx=20,
                              pady=10,
                              command=self.exit_application,
                              cursor="hand2")
        finish_btn.pack(side=tk.LEFT, padx=10)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (400)
    y = (root.winfo_screenheight() // 2) - (350)
    root.geometry(f"+{x}+{y}")
    
    # Set minimum window size
    root.minsize(700, 600)
    
    app = SVMGradePredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
import pandas as pd

FILE_PATH = "Student_performance_data _.csv"

def checkMissingValue():
    
    df = pd.read_csv(FILE_PATH)
    missing_counts = df.isnull().sum()
    total_missing_count = missing_counts.sum()

    print("Missing values in each column")
    print(missing_counts)

    if total_missing_count > 0:
        print(f"\nTotal missing values: {total_missing_count}")
        print("Removing rows with missing values...")
        df = df.dropna()
        print(f"New dataset shape after removal: {df.shape}")
    else:
        print("\nNo missing values found in the dataset.")

    return missing_counts  

def main():

    cleaned_df = checkMissingValue()
    # Optional: save cleaned data
    cleaned_df.to_csv("Cleaned_StudentsPerformance.csv", index=False)
    print("Cleaned data saved.")
    

if __name__ == "__main__":
    main()

# edit
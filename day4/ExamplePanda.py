# Step 1: Import necessary libraries
import pandas as pd

# Step 2: Load the CSV data into a DataFrame
df = pd.read_csv("data.csv")  # Replace with your actual file name

# Step 3: Quick exploration
print("🔹 Data Overview")
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Handling missing data
df = df.dropna(subset=["Age", "City"])  # Drop rows where Age or City is missing
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())  # Fill missing Salary with average

# Step 5: Filtering rows with conditions (use parentheses!)
filtered_df = df[(df["Age"] > 25) & (df["City"] == "New York")].copy()

# Step 6: Modify safely
filtered_df["IsSenior"] = filtered_df["Age"] > 40

# Step 7: Grouping and Aggregation
avg_salary_by_dept = df.groupby("Department")["Salary"].mean()
print("\n🔹 Average Salary by Department:\n", avg_salary_by_dept)

# Step 8: Combined filtering + grouping
high_earners = df[df["Salary"] > 50000]
salary_by_city = high_earners.groupby("City")["Salary"].mean()
print("\n🔹 High Earners' Average Salary by City:\n", salary_by_city)

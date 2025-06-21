import pandas as pd

# Step 1: Load the data
df = pd.read_csv('sales_data.csv')
print("Original Data:")
print(df.head(10))
print("nInfo:")
print(df.info())

# Step 2: Detect missing values
print("nMissing values per column:")
print(df.isnull().sum())

# Step 3: Handle missing values
# Fill missing numerical values with mean
df['Quantity'].fillna(df['Quantity'].mean(), inplace=True)
df['Price'].fillna(df['Price'].mean(), inplace=True)

# Fill missing categorical values with 'Unknown'
df['Customer Name'].fillna('Unknown', inplace=True)
df['Region'].fillna('Unknown', inplace=True)

# Fill missing dates with mode
df['Sale Date'].fillna(df['Sale Date'].mode()[0], inplace=True)

print("nData after filling missing values:")
print(df.head(10))

# Step 4: Remove duplicates
print("nNumber of duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Step 5: Rename columns to lowercase and replace spaces with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
print("nColumns after renaming:")
print(df.columns)

# Step 6: Summarize the cleaned data
print("nSummary statistics:")
print(df.describe())

print("nTotal quantity and average price per region:")
region_summary = df.groupby('region').agg({'quantity':'sum', 'price':'mean'})
print(region_summary)

print("nTop 3 products by quantity sold:")
product_summary = df.groupby('product')['quantity'].sum().sort_values(ascending=False).head(3)
print(product_summary)

# Bonus: Create total_sale column
df['total_sale'] = df['quantity'] * df['price']

print("nDate with highest total sales:")
date_sales = df.groupby('sale_date')['total_sale'].sum()
print(date_sales.sort_values(ascending=False).head(1))

# Step 7: Save cleaned data
df.to_csv('sales_data_cleaned.csv', index=False)

print("nCleaned data saved to 'sales_data_cleaned.csv'")



# ------------------- OUTPUT -------------------

# Original Data:
#   Customer Name   Product  Quantity  Price   Sale Date Region
# 0         Alice  Keyboard       5.0   25.0  2023-05-01  North
# 1           Bob     Mouse       3.0    NaN  2023-05-03  South
# 2           NaN   Monitor       2.0  150.0  2023-05-01   East
# 3         David  Keyboard       5.0   25.0  2023-05-01  North
# 4         Alice  Keyboard       5.0   25.0  2023-05-01  North
# 5           Eve    Laptop       NaN  900.0         NaN   West
# 6         Frank     Mouse       1.0   20.0  2023-05-04    NaN
# 7         Grace   Monitor       2.0  150.0  2023-05-02   East
# 8           Bob     Mouse       3.0    NaN  2023-05-03  South
# 9         Henry    Laptop       1.0  900.0  2023-05-05   West

# Missing values per column:
# Customer Name    1
# Product          0
# Quantity         1
# Price            2
# Sale Date        1
# Region           1

# Data after filling missing values:
#   Customer Name   Product  Quantity       Price   Sale Date   Region
# 0         Alice  Keyboard       5.0   25.000000  2023-05-01    North
# 1           Bob     Mouse       3.0  246.666667  2023-05-03    South
# 2       Unknown   Monitor       2.0  150.000000  2023-05-01     East
# 3         David  Keyboard       5.0   25.000000  2023-05-01    North
# 4         Alice  Keyboard       5.0   25.000000  2023-05-01    North
# 5           Eve    Laptop       3.1  900.000000  2023-05-01     West
# 6         Frank     Mouse       1.0   20.000000  2023-05-04  Unknown
# 7         Grace   Monitor       2.0  150.000000  2023-05-02     East
# 8           Bob     Mouse       3.0  246.666667  2023-05-03    South
# 9         Henry    Laptop       1.0  900.000000  2023-05-05     West

# Number of duplicate rows: 2

# Columns after renaming:
# Index(['customer_name', 'product', 'quantity', 'price', 'sale_date', 'region'], dtype='object')

# Summary statistics:
#         quantity       price
# count   9.000000    9.000000
# mean    2.900000  271.296296
# std     1.537856  364.983616
# min     1.000000   20.000000
# 25%     2.000000   25.000000
# 50%     3.000000  150.000000
# 75%     4.000000  246.666667
# max     5.000000  900.000000

# Total quantity and average price per region:
#           quantity       price
# region
# East           4.0  150.000000
# North         14.0   25.000000
# South          3.0  246.666667
# Unknown        1.0   20.000000
# West           4.1  900.000000

# Top 3 products by quantity sold:
# product
# Keyboard    14.0
# Laptop       4.1
# Monitor      4.0
# Name: quantity, dtype: float64

# Date with highest total sales:
# sale_date
# 2023-05-01    3440.0
# Name: total_sale, dtype: float64

# Cleaned data saved to 'sales_data_cleaned.csv'

# -------------------------------------------------------




# ------------------- Summary Report -------------------

# 1. Missing Data Handling:
# - 'Quantity' and 'Price' columns had missing values, which were filled using the mean of their respective columns.
# - 'Customer Name' and 'Region' missing entries were filled with 'Unknown'.
# - 'Sale Date' missing value was filled with the most frequent date (mode).

# 2. Duplicate Removal:
# - Found and removed 2 duplicate rows to ensure data accuracy.

# 3. Column Formatting:
# - All column names were converted to lowercase and spaces were replaced with underscores for consistency.

# 4. New Column:
# - Added a 'total_sale' column calculated as quantity * price.

# 5. Insights:
# - Top product by quantity sold: 'Keyboard' (14 units).
# - Date with highest total sales: '2023-05-01' with ₹3440.00 in total sales.
# - Region-wise summary provided with total quantity and average price.

# Final cleaned dataset was saved to 'sales_data_cleaned.csv'.

# -------------------------------------------------------

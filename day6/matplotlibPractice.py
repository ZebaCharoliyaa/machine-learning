

# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample Weather Dataset
# data = {
#     'Day': list(range(1, 11)),
#     'Temperature': [30, 32, 31, 29, 28, 27, 30, 31, 33, 34],
#     'Rainfall': [5, 10, 3, 0, 0, 15, 20, 5, 2, 0],
#     'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Cloudy', 'Sunny']
# }

# df = pd.DataFrame(data)

# # Line Plot - Temperature over Days
# plt.figure(figsize=(8, 4))
# plt.plot(df['Day'], df['Temperature'], marker='o', color='orange')
# plt.title('Temperature Over Days')
# plt.xlabel('Day')
# plt.ylabel('Temperature (°C)')
# plt.grid(True)
# plt.savefig('temperature_line_chart.png')
# plt.close()

# # Bar Chart - Average Rainfall (grouped by weather type or fake monthly data)
# # Assuming 3 months to demonstrate monthly average
# monthly_data = {
#     'Month': ['Jan', 'Feb', 'Mar'],
#     'Avg_Rainfall': [20, 35, 15]
# }
# monthly_df = pd.DataFrame(monthly_data)

# plt.figure(figsize=(6, 4))
# plt.bar(monthly_df['Month'], monthly_df['Avg_Rainfall'], color='blue')
# plt.title('Average Rainfall Per Month')
# plt.xlabel('Month')
# plt.ylabel('Rainfall (mm)')
# plt.savefig('average_rainfall_bar_chart.png')
# plt.close()

# # Scatter Plot - Temperature vs Rainfall
# plt.figure(figsize=(6, 4))
# plt.scatter(df['Temperature'], df['Rainfall'], color='green')
# plt.title('Temperature vs Rainfall')
# plt.xlabel('Temperature (°C)')
# plt.ylabel('Rainfall (mm)')
# plt.savefig('temp_vs_rainfall_scatter.png')
# plt.close()

# # Pie Chart - Weather Type Distribution
# weather_counts = df['Weather'].value_counts()

# plt.figure(figsize=(6, 6))
# plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90, colors=['gold', 'skyblue', 'lightgrey'])
# plt.title('Weather Type Distribution')
# plt.savefig('weather_type_pie_chart.png')
# plt.close()



import pandas as pd
import matplotlib.pyplot as plt

# Sample Weather Dataset
data = {
    'Day': list(range(1, 11)),
    'Temperature': [30, 32, 31, 29, 28, 27, 30, 31, 33, 34],
    'Rainfall': [5, 10, 3, 0, 0, 15, 20, 5, 2, 0],
    'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Cloudy', 'Sunny']
}

df = pd.DataFrame(data)

# Line Plot - Temperature over Days
plt.figure(figsize=(8, 4))
plt.plot(df['Day'], df['Temperature'], marker='o', color='orange')
plt.title('Temperature Over Days')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.savefig('temperature_line_chart.png')
plt.show()
# plt.close()

# Bar Chart - Average Rainfall (grouped by weather type or fake monthly data)
# Assuming 3 months to demonstrate monthly average
monthly_data = {
    'Month': ['Jan', 'Feb', 'Mar'],
    'Avg_Rainfall': [20, 35, 15]
}
monthly_df = pd.DataFrame(monthly_data)

plt.figure(figsize=(6, 4))
plt.bar(monthly_df['Month'], monthly_df['Avg_Rainfall'], color='blue')
plt.title('Average Rainfall Per Month')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.savefig('average_rainfall_bar_chart.png')
plt.show()

# plt.close()

# Scatter Plot - Temperature vs Rainfall
plt.figure(figsize=(6, 4))
plt.scatter(df['Temperature'], df['Rainfall'], color='green')
plt.title('Temperature vs Rainfall')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.savefig('temp_vs_rainfall_scatter.png')
plt.show()

# plt.close()

# Pie Chart - Weather Type Distribution
weather_counts = df['Weather'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90, colors=['gold', 'skyblue', 'lightgrey'])
plt.title('Weather Type Distribution')
plt.savefig('weather_type_pie_chart.png')
plt.show()

# plt.close()

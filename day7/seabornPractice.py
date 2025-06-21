import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("pastel")

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Quick look at data
titanic.head()

plt.figure(figsize=(8, 5))
sns.histplot(data=titanic, x='age', kde=True, color='skyblue')
plt.title('Distribution of Passenger Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('age_distribution.png')
plt.show()

# fare distribution 
plt.figure(figsize=(8, 5))
sns.kdeplot(data=titanic, x='fare', fill=True, color='orange')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.savefig('fare_distribution.png')
plt.show()

#  Relationship Plots
# Age vs Fare
plt.figure(figsize=(8, 5))
sns.scatterplot(data=titanic, x='age', y='fare', hue='class')
plt.title('Age vs Fare by Class')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Class')
plt.savefig('age_vs_fare.png')
plt.show()

#  Age vs Survival
plt.figure(figsize=(8, 5))
sns.lineplot(data=titanic, x='age', y='survived', errorbar=None)
plt.title('Age vs Survival Rate')
plt.xlabel('Age')
plt.ylabel('Survival Probability')
plt.savefig('age_vs_survival.png')
plt.show()

# Categorical Plots
# Survival Count by Gender

plt.figure(figsize=(8,5))
sns.countplot(data=titanic, x='sex', hue='survived', palette='Set2')
plt.title('Survival count by gender')
plt.xlabel('gender')
plt.ylabel('Count')
plt.savefig('survival by gender.png')
plt.show()

# bar plot
#  Average Fare by Class
plt.figure(figsize=(8, 5))
sns.barplot(data=titanic, hue='class', y='fare',palette='coolwarm',legend='auto')
plt.title('Average fare by clas')
plt.xlabel('class')
plt.ylabel('Fare')
plt.savefig('avg_fare_by_class.png')
plt.show()
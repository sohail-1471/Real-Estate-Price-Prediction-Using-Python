
""" Real Estate Price Prediction """

# let’s get started with the task of building a hybrid machine learning model
# by importing the necessary Python libraries and the dataset:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:/Data Science/Data Science projects/Machine Learning Projects/Supervised Learning- Classification & Regression/6. Real Estate Price Prediction using Linear regression/Real_Estate.csv", encoding='latin-1')
pd.set_option('display.max_columns', None)
# Let's have a look at first ten rows of the dataset:
print(data.head(10))

# Display the info
data_info = data.info()

# Let's have a look if data contains any null values or not:
print(data.isnull().sum())

# Now Let's have a look at the descriptive statistics of the data
print(data.describe())

# Visualization Section

#1. Now Let's have a look at the histograms of all numerical features

# Set the aesthetic style of the plots
sns.set_style('darkgrid')

# Create histograms for the numerical columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Histogram of Real Estate Data', fontsize=16, fontweight='bold')

cols = ['House age','Distance to the nearest MRT station','Number of convenience stores','Latitude','Longitude','House price of unit area']

for i, col in enumerate(cols):
    sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#2. Scatter plots to observe the relationship with house price
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
fig.suptitle('Scatter plots with house price of unit area', fontsize=16, fontweight='bold')

# Scatter plot for each variable against the house price
sns.scatterplot(data=data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#3. Now Let's perform a correlation analysis to quantify the relationships between these variables:
# Correlation Matrix
data_corr = data[['House age','Distance to the nearest MRT station','Number of convenience stores','Latitude','Longitude','House price of unit area']]
correlation_matrix = data_corr.corr()

# Plotting the Correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
print(correlation_matrix)

# Modeling
#1. Regression Model
# Let's build a regression model to predict the real estate prices by using the linear regression algorithm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting Features and Target Variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'
X = data[features]
y = data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions using linear regression model
y_pred = model.predict(X_test)

# Defining a threshold for outliers
threshold = 10000  # You can adjust this value based on your criteria

# Calculating the residuals (errors)
residuals = abs(y_test - y_pred)

# Visualise the actual vs predicted values
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test,y=y_pred, alpha=0.6)
sns.lineplot(x=[y_test.min(), y_test.max()],y=[y_test.min(), y_test.max()], color='green')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()












import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

st.header("Housing Price Estimator :house_buildings:")

# Load the data using Pandas
data = pd.read_csv("data.csv")  # Replace 'data.csv' with the path to your dataset

# Extract the input features (X) and the target variable (Y)
X = data[['House_Size']]  # Assuming 'House_Size' is the column name for the independent variable
Y = data['House_Price']  # Assuming 'House_Price' is the column name for the dependent variable

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Plot the data points
plt.scatter(X, Y, marker="+", color='black', label='Data Points')

# Plot the regression line
plt.plot(X, model.predict(X), color='red', linewidth=1, label='Regression Line')

# Create a formatter function to format the y-axis tick labels with commas
formatter = ticker.StrMethodFormatter("{x:,.0f}")
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True)

# Print the learned coefficients
print("Intercept:", model.intercept_)  # Intercept (b) value
print("Coefficient:", model.coef_)  # Slope (m) value(s)

# Add labels and title to the plot
plt.xlabel('House Size (m^2)')
plt.ylabel('House Price ($)')
plt.title('Bivariate Linear Regression Model for housing price')

# Add a legend
plt.legend()

# Display plot on app
st.pyplot(plt)

st.divider()

# Summary
st.subheader("Summary")
st.write("This bivariate linear regression model aims to provide users an estimate of a house price based on a dataset of 100 data points. "+
         "See data below.")

data

st.divider()
st.subheader("Prediction")
st.caption("Value entered here will be used to provide an estimated price of a house based on the training data provided to the OLS model.")

# Capture user input for prediction
input_house_size = st.slider("House Size Input", min_value=65, max_value=200, value=120)

prediction = model.predict(pd.DataFrame({'House_Size': [input_house_size]}))
prediction_formatted = f'${round(prediction[0]):,}'

st.write("The predicted price of a house, based on the OLS model is: "+ prediction_formatted)

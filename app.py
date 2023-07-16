import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

# Import the wrapper function from your package
from streamlit_custom_components import st_custom_slider

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
st.write("This bivariate linear regression model aims to provide users an estimate of a house price given a specific house size in square metres. " +  
         "A dataset of 100 dummy data points generated by ChatGPT has been used for this model which consists of an independent variable (House Size), " +
         "as well as a dependent variable (House Price).")
st.write("Mathematically, the model is expressed by the equation below")
st.latex("Y = b_0 + b_1X")
st.write("The constant term and cofficient of X represents the line-intercept and the slope of the plot respectively, where the more interesting bit " +
         "being the cofficient term which indicates the amount of increment in housing price given a unit increase in house size. The model has been trained " +
         "on the dataset below which generated the following relationship.")
st.latex("Y = -55426.14 + 3512X")

data

st.divider()
st.subheader("Prediction")
st.caption("Value entered here will be used to provide an estimated price of a house based on the training data provided to the OLS model.")

# Capture user input for prediction
input_house_size = st_custom_slider()

prediction = model.predict(pd.DataFrame({'House_Size': [input_house_size]}))
prediction_formatted = f'${round(prediction[0]):,}'

st.write("The predicted price of a house, based on the OLS model is: "+ prediction_formatted)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.header("Ordinary Least Squares")

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
plt.scatter(X, Y, color='blue', label='Data')

# Plot the regression line
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

# Print the learned coefficients
print("Intercept:", model.intercept_)  # Intercept (b) value
print("Coefficient:", model.coef_)  # Slope (m) value(s)

# Add labels and title to the plot
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Linear Regression')

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

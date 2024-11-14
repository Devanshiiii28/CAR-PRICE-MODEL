import streamlit as st

#Title
# Streamlit UI
st.title('Car Price Prediction')
#HEADER
st.header("Features")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Save the trained model to a pickle file
def save_model():
    # Example dataset (This can be replaced with your actual data)
    data = {
        'name': ['Maruti Suzuki Swift', 'Hyundai i20', 'Honda City', 'Toyota Corolla', 'Ford Fiesta'],
        'company': ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford'],
        'year': [2019, 2020, 2018, 2021, 2017],
        'kms_driven': [10000, 12000, 5000, 3000, 20000],
        'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
        'price': [800000, 900000, 1000000, 1200000, 700000]  # Example target variable
    }

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Preprocess the data (encode categorical features)
    label_encoder_company = LabelEncoder()
    df['company_encoded'] = label_encoder_company.fit_transform(df['company'])
    
    label_encoder_fuel = LabelEncoder()
    df['fuel_type_encoded'] = label_encoder_fuel.fit_transform(df['fuel_type'])

    # Prepare features and target variable
    X = df[['year', 'kms_driven', 'company_encoded', 'fuel_type_encoded']]
    y = df['price']

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model and label encoders using pickle
    with open("LinearRegressionModel.pkl", "wb") as file:
        pickle.dump(model, file)
    
    with open("LabelEncoder_Company.pkl", "wb") as file:
        pickle.dump(label_encoder_company, file)

    with open("LabelEncoder_Fuel.pkl", "wb") as file:
        pickle.dump(label_encoder_fuel, file)

    st.write("Model saved successfully!")

# Load the model and label encoders from pickle files
def load_model():
    with open("LinearRegressionModel.pkl", "rb") as file:
        model = pickle.load(file)
    
    with open("LabelEncoder_Company.pkl", "rb") as file:
        label_encoder_company = pickle.load(file)
    
    with open("LabelEncoder_Fuel.pkl", "rb") as file:
        label_encoder_fuel = pickle.load(file)

    return model, label_encoder_company, label_encoder_fuel



# Input fields for user input
name = st.text_input('Car Name', 'Maruti Suzuki Swift')
company = st.selectbox('Company', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford'])
year = st.number_input('Year', 2019, 2023, 2020)
kms_driven = st.number_input('KMS Driven', 0, 200000, 10000)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])

# Save model button
if st.button('Train and Save Model'):
    save_model()

# When the user presses the 'Predict' button
if st.button('Predict Price'):
    try:
        # Load the trained model and label encoders
        model, label_encoder_company, label_encoder_fuel = load_model()

        # Encode categorical features for prediction
        # Handle unseen categories gracefully by assigning a default value (e.g., -1 or a specific fallback)
        try:
            company_encoded = label_encoder_company.transform([company])[0]
        except ValueError:
            # If the company is unseen, we can assign a default value or handle it in another way
            company_encoded = -1  # Default value for unseen category (this could be modified based on your needs)

        try:
            fuel_type_encoded = label_encoder_fuel.transform([fuel_type])[0]
        except ValueError:
            # If the fuel type is unseen, we can assign a default value or handle it in another way
            fuel_type_encoded = -1  # Default value for unseen category (this could be modified based on your needs)

        # Prepare the input features for prediction
        input_features = np.array([[year, kms_driven, company_encoded, fuel_type_encoded]])

        # Predict the price using the loaded model
        predicted_price = model.predict(input_features)[0]

        # Display the result
        st.write(f'Predicted Price for {name} is: ₹{predicted_price:,.2f}')

    except Exception as e:
        st.write(f"Error: {e}. Please train the model first using 'Train and Save Model'.")

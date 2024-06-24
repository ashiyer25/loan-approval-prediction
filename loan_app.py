import streamlit as st
import pandas as pd
import numpy as np
import pickle
from Orange.data import *

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Predict'])

def predictions(gender, married, dependents, education, self_employed, applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area):
    # Define the domain
    gender_var = DiscreteVariable("Gender", values=["Male", "Female"])
    married_var = DiscreteVariable("Married", values=["Yes", "No"])
    dependents_var = DiscreteVariable("Dependents", values=["0", "1", "2", "3+"])
    education_var = DiscreteVariable("Education", values=["Graduate", "Not Graduate"])
    self_employed_var = DiscreteVariable("Self_Employed", values=["Yes", "No"])
    applicantincome_var = ContinuousVariable("ApplicantIncome")
    coapplicantincome_var = ContinuousVariable("CoApplicantIncome")
    loanamount_var = ContinuousVariable("LoanAmount")
    loan_amount_term_var = ContinuousVariable("Loan_Amount_Term")
    credit_history_var = DiscreteVariable("Credit_History", values=["0.0", "1.0"])
    property_area_var = DiscreteVariable("Property_Area", values=["Urban", "Semiurban", "Rural"])

    domain = Domain([gender_var, married_var, dependents_var, education_var, self_employed_var,
                     applicantincome_var, coapplicantincome_var, loanamount_var,
                     loan_amount_term_var, credit_history_var, property_area_var])

    # Create a data instance
    data = Table(domain, [[gender, married, dependents, education, self_employed,
                           applicantincome, coapplicantincome, loanamount,
                           loan_amount_term, credit_history, property_area]])

    # Load the model
    with open("loan_model.pkcls", "rb") as f:
        model_loaded = pickle.load(f)

    # Make prediction
    output = model_loaded(data)
    preds = model_loaded.domain.class_var.str_val(output)
    if preds == "Y":
        preds = "Yes the loan is approved"
    else:
        preds = "No the loan is not approved"
    st.success(preds)
    return preds

if app_mode == 'Home':
    st.title('Loan Approval Prediction')
    st.image('image.jpg')
    st.markdown('''
    Welcome to the Loan Approval Prediction App. This application uses a machine learning model to predict whether a loan will be approved based on the applicant's details.
    The following are the steps to utilise this tool:--
    
    1. If accessing through PC/Laptop device - On your Left hand side of the page, under "Select Page", choose "Predict" via the drop down list. 
    2. If accessing through a mobile device, head to the top right corner of the screen, tap on the ">" button, from the pane that shows up, follow the same step as the previous one.
    3. Navigate to the prediction page using the sidebar to input your details and get a loan approval prediction.
    4. Click on "Predict" button at the end of the page to receive results.
    ''')

elif app_mode == 'Predict':
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    applicantincome = st.slider("Applicant Income:", min_value=0, max_value=100000)
    coapplicantincome = st.sidebar.number_input("Coapplicant Income:", min_value=0, max_value=100000)
    loanamount = st.sidebar.number_input('Loan Amount', min_value=10000, max_value=1000000)
    loan_amount_term = st.sidebar.number_input('Loan Amount Term', min_value=0, max_value=500)
    credit_history = st.sidebar.selectbox('Credit History', (1.0, 0.0))
    property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    # Trigger prediction
    if st.button("Predict"):
        predictions(gender, married, dependents, education, self_employed, applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area)

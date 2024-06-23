import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from Orange.data import *

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict'])

def predictions(gender, married, dependents, education, self_employed, applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area):
    
    gender = DiscreteVariable("gender",values=["Male","Female"])
    married = DiscreteVariable("married",values=["Yes", "No"])
    dependents = DiscreteVariable("dependents", values=["1","2","3"])
    education = DiscreteVariable("education", values=["Graduate", "Not Graduate"])
    applicantincome= ContinuousVariable("applicantincome")
    coapplicantincome=ContinuousVariable("coapplicantincome")
    loanamount=ContinuousVariable("loanamount")
    loan_amount_term=ContinuousVariable("loan_amount_term")
    credit_history=DiscreteVariable("credit_history",values=["0","1"])
    property_area=DiscreteVariable("property_area",values=["Rural","Urban", "Semiurban"])
    
    
    
    
    domain = Domain([gender, married, dependents, education, self_employed , applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area])   
    data=Table(domain,[[gender, married, dependents, education, self_employed , applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area]])
    
    # Load the model
    with open("naive.pkcls", "rb") as f:
        model_loaded = pickle.load(f)
       
       
       
       
    if st.button("Predict"):
        
        output = model_loaded(data)
        preds = model_loaded.domain.class_var(output)
        st.success(preds)
        
        return preds
        
if app_mode == 'Home': 
    st.title('Loan Approval Prediction')
    st.image('image.jpg')
    st.markdown(''' Welcome to the Loan Approval Prediction App. This application uses a machine learning model to predict whether a loan will be approved based on the applicant's details.
    The following are the steps to utilise this tool:--
    
    1. If accessing through PC/Laptop device - On your Left hand side of the page, 
    under "Select Page", choose "Predict" via the drop down list. 
    
    2. If accessing through a mobile device, head to the top right corner of the 
    screen, tap on the ">" button, from the pane that shows up, follow the 
    same step as the previous one.
    
    3. Navigate to the prediction page using the sidebar to input your details and get a loan approval prediction.
    
    
    4. Click on "Predict" button at the end of the page to receive results. ''')
    
    
elif app_mode == 'Predict':
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
    applicantincome = st.slider("Applicant Income:", min_value=0)
    coapplicantincome = st.sidebar.number_input("Coapplicant Income:", min_value=0)
    loanamount = st.sidebar.number_input('Loan Amount', min_value=10000, max_value=1000000)
    loan_amount_term = st.sidebar.number_input('Loan Amount Term', min_value=0, max_value=500)
    credit_history = st.sidebar.selectbox('Credit History', (1.0, 0.0))
    property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))    
        
        
    
    predictions(gender,married, dependents, education, self_employed, applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area)
     
        

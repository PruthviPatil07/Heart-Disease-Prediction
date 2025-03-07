import streamlit as st 
import pandas as pd 
import joblib
import numpy as np 

#age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal

st.header('Heart Disease Prediction')

import pandas as pd
import streamlit as st

# Get user inputs from Streamlit
actual_age = st.number_input('Enter Age')
actual_sex = st.number_input('Enter Sex')
actual_cp = st.number_input('Enter CP')
actual_trestbps = st.number_input('Enter Trestbps')
actual_chol = st.number_input('Enter Chol')
actual_fbs = st.number_input('Enter FBS')
actual_restecg = st.number_input('Enter Restecg')
actual_thalach = st.number_input('Enter Thalach')
actual_exang = st.number_input('Enter Exang')
actual_oldpeak = st.number_input('Enter Oldpeak')
actual_slope = st.number_input('Enter Slope')
actual_ca = st.number_input('Enter CA')
actual_thal = st.number_input('Enter Thal')

# Create a DataFrame 

df = pd.DataFrame({
    'age': [actual_age],
    'sex': [actual_sex],
    'cp': [actual_cp],
    'trestbps': [actual_trestbps],
    'chol': [actual_chol],
    'fbs': [actual_fbs],
    'restecg': [actual_restecg],
    'thalach': [actual_thalach],
    'exang': [actual_exang],
    'oldpeak': [actual_oldpeak],
    'slope': [actual_slope],
    'ca': [actual_ca],
    'thal': [actual_thal]
})

model = joblib.load(r'C:\Users\PRUTHVIRAJ\OneDrive\Desktop\Bootcamp\models\best_model')

pred = model.predict(df)

submit = st.button('Enter to see the prediction')
if submit:
    st.subheader('The Prediction Is...')
    st.write(pred)
    
if pred == 0:
    st.write('You Have No Disease')
else:
    st.write('You Have Heart Disease')
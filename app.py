import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import pickle as pkl
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

model=pkl.load(open('IPML.pkl','rb'))
st.header('Insurance Premium Predictor')

sex=st.selectbox('Select Gender',['Male', 'Female'])
region=st.selectbox('Select region',['southeast', 'southwest','northeast','northwest'])
smoker=st.selectbox('Are you Smoker?',['Yes', 'No'])
age=st.slider('Select Age',3,100)
bmi=st.slider('Select BMI',10,100)
children=st.slider('How many children you have',0,10)

if sex=='female':
    sex=0
else:
    sex=1
if region=='southeast':
    region=0
elif region=='southwest':
    region=1
elif region=='northeast':
    region=2
else:
    region=3
if smoker=='yes':
    smoker=1
else:
    smoker=0

input_data=(age,sex,bmi,children,smoker,region)
input_data=np.asarray(input_data)
input_data=input_data.reshape(1,-1)
if st.button('predict'):
    predicted_premium=model.predict(input_data)
    display_string='Insurance Premium will be '+ 'Rs. '+ str(round(predicted_premium[0]))+'.'
    st.markdown(display_string)


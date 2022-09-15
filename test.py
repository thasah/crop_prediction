import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.write("""
## Crop Prediction Web App
 
""")

st.sidebar.header('User input parameters')

def user_input_features():
    N = st.sidebar.slider('Nitrogen', 0,140,40)
    P = st.sidebar.slider('Phosporos', 5,145,50)
    K = st.sidebar.slider('Potassium',5,205,50)
    temperature = st.sidebar.slider('Temperature', 8,45,29)
    humidity = st.sidebar.slider('Humidity', 14,100,50)
    ph = st.sidebar.slider('pH Value', 3,10,7)
    rainfall = st.sidebar.slider('Rainfall',20,300,200)
    data = {'Nitrogen':N,
            'Phosporos':P,
            'Potassium':K,
            'Temperature':temperature,
            'Humidity':humidity,
            'pH Value': ph,
            'Rainfall':rainfall}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

crop = pd.read_csv("Crop_recommendation.csv")
X = crop.loc[:, 'N':'rainfall']
Y = df.loc[:,'label']

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index numbers')
st.write(crop['label'].unique())

st.subheader('Prediction')
st.write(crop['label'].unique(prediction))

st.subheader('Prediction Probability')
st.write(prediction_proba)
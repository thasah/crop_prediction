import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Crop Prediction App 

This app predicts the crops when given user inputs

""")

st.sidebar.header('User input feature')
st.sidebar.markdown("""
[Example csv input file](sample_inputfile.csv)
""")
uploaded_file = st.sidebar.file_uploader('Upload your input file', type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        N = st.sidebar.slider('Nitrogen', 0,140,40)
        P = st.sidebar.slider('Phosporos', 5,145,50)
        K = st.sidebar.slider('Potassium',5,205,50)
        temperature = st.sidebar.slider('Temperature', 8,45,29)
        humidity = st.sidebar.slider('Humidity', 14,100,50)
        ph = st.sidebar.slider('pH Value', 3,10,7)
        rainfall = st.sidebar.slider('Rainfall',20,300,200)
        data = {'N':N,
                'P':P,
                'K':K,
                'temperature':temperature,
                'humidity':humidity,
                'ph': ph,
                'rainfall':rainfall}
        features = pd.DataFrame(data,index=[0])
        return features
    input_df = user_input_features()

crops_raw = pd.read_csv('Crop_recommendation.csv')
crops = crops_raw.drop(columns=['label'])
df = pd.concat([input_df,crops], axis=0)

df=df[:1]

st.subheader('User Input Parameters')

if uploaded_file is not None: 
    st.write(df)
else:
    st.write('Change the parameters or upload input values file')
    st.write(df)

load_clf = pickle.load(open('crops_clf.pkl','rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
crop_types = np.array(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'])
st.write(crop_types[prediction])

st.subheader('Prediction probability')
st.write(prediction_proba)
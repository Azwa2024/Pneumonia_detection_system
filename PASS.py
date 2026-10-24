import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('pneumonia_model.pkl')
train_columns = joblib.load("columns.pkl")
st.set_page_config(page_title=': Pneumonia Assessment and Screening System ', layout='centered')
st.title('🫁 Pneumonia Assessment and Screening System ')
st.write('Enter patient medical details to predict pneumonia risk.')
age = st.number_input('Age', min_value=0, max_value=100, value=0, step=1)

gender = st.selectbox('Gender', ['male', 'female'])
cough=st.selectbox('Cough',['None','Bloody','Dry','Wet'])
fever=st.selectbox('Fever',['None','High','Moderate','Low'])
shortness=st.selectbox('Shortness of Breath',['None','Severe','Moderate','Mild'])
chest=st.selectbox('Chest pain',['None','Severe','Moderate','Mild'])
fatigue=st.selectbox('Fatigue',['Unknown','Severe','Moderate','Mild'])
confusion = st.selectbox('Confusion', ['yes', 'no', 'Unknown'])

crackles = st.selectbox('Crackles: Unusual crackling sounds in the lungs', ['yes', 'no'])
sputum = st.selectbox('Sputum Color',['yellow', 'green', 'clear', 'bloody', 'Unknown'])
oxygen=st.number_input('Oxygen_Saturation', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
temp=st.number_input('Temperature', min_value=0.0, max_value=300.0, value=0.0, step=0.1)

input_data={
     'Age':[age],
    
    'Oxygen_saturation':[oxygen],

    'Temperature':[temp] ,
    
     'Gender_m':[1 if gender=='male' else 0],
     'Gender_f':[1 if gender=='female' else 0],
     
     'Cough_none':[1 if cough=='None' else 0],
    'Cough_bloody':[1 if cough=='Bloody' else 0],
    'Cough_dry':[1 if cough=='Dry' else 0],
    'Cough_wet':[1 if cough=='Wet' else 0],
   
    'Fever_none':[1 if fever=='None' else 0],
    'Fever_high':[1 if fever=='High' else 0],
    'Fever_low':[1 if fever=='Low' else 0],
    'Fever_moderate':[1 if fever=='Moderate' else 0],

    'Shortness_of_breath_none':[1 if shortness=='None' else 0],
     'Shortness_of_breath_mild':[1 if shortness=='Mild' else 0],
    'Shortness_of_breath_moderate':[1 if shortness=='Moderate' else 0],
    'Shortness_of_breath_severe':[1 if shortness=='Severe' else 0],

     'Chest_pain_none':[1 if chest=='None' else 0],
    'Chest_pain_mild':[1 if chest=='Mild' else 0],
    'Chest_pain_moderate':[1 if chest=='Moderate' else 0],
    'Chest_pain_severe':[1 if chest=='Severe' else 0],

    
     'Fatigue_none':[1 if fatigue=='Unknown' else 0],
    'Fatigue_mild':[1 if fatigue=='Mild' else 0],
    'Fatigue_moderate':[1 if fatigue=='Moderate' else 0],
    'Fatigue_severe':[1 if fatigue=='Severe' else 0],

    'Confusion_unknown':[1 if confusion=='Unknown' else 0],
    'Confusion_no':[1 if confusion=='no' else 0],
    'Confusion_yes':[1 if confusion=='yes' else 0],

    
    'Crackles_no':[1 if crackles=='no' else 0],
    'Crackles_yes':[1 if crackles=='yes' else 0],

    'Sputum_color_unknown':[1 if sputum=='Unknown' else 0],
    'Sputum_color_bloody':[1 if sputum=='bloody' else 0],
    'Sputum_color_clear':[1 if sputum=='clear' else 0],
    'Sputum_color_green':[1 if sputum=='green' else 0],
    'Sputum_color_yellow':[1 if sputum=='yellow' else 0],

   
       }
input_df=pd.DataFrame(input_data)
input_df = input_df.reindex(columns=train_columns, fill_value=0)

if st.button('Predict Pneumonia Risk'):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader('Prediction Result')

    if prediction == 1:
        st.error('⚠️ High Chance of Pneumonia Detected')
    else:
        st.success('✅ Low Chance of Pneumonia')

    st.write(f'### Pneumonia Probability: {probability*100:.2f}%')
    st.progress(float(probability))

    

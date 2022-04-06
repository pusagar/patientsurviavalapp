import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder,labelencoder
import tensorflow as tf
from tensorflow import keras
#import load_model
#from load_model import get_model

#model = get_model(model_path = r'model/keras_model.h5')
#model = joblib.load(r'model/keras_model.h5')
model = keras.models.load_model(r'model/keras_model.h5')

st.set_page_config(page_title="Patient Survival App",page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_apache_2_bodysystem = ['Sepsis','Respiratory','Metabolic','Cardiovascular','Trauma',
       'Neurological','Gastrointestinal','Genitourinary','Hematological','Musculoskeletal/Skin','Gynecological']

#options_apache_3j_bodysystem = ['Sepsis', 'Respiratory', 'Metabolic', 'Cardiovascular', 'Trauma',
#       'Neurological', 'Gastrointestinal', 'Genitourinary', 
#       'Hematological', 'Musculoskeletal/Skin', 'Gynecological']
      

features = ['apache_2_bodysystem','solid_tumor_with_metastasis','lymphoma','leukemia',
           'immunosuppression','hepatic_failure','diabetes_mellitus','cirrhosis','aids']


st.markdown("<h1 style='text-align: center;'> Patient Survival App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
                
        apache_2_bodysystem = st.selectbox("Select State Factor: ", options=options_apache_2_bodysystem)
        #apache_3j_bodysystem = st.selectbox("Select State Factor: ", options=options_apache_3j_bodysystem)
        solid_tumor_with_metastasis = st.slider("Does the patient have solid tumor with metastasis: ", 0.00, 1.00, value=0.,step = 0.05, format="%f")
        lymphoma = st.slider("Does the patient have lymphoma : ", 0.00, 1.00, value=0., step = 0.05, format="%f")
        leukemia = st.slider("Does the patient have leukemia : ", 0.00, 1.00, value=0.,step = 0.05, format="%f")
        immunosuppression = st.slider("Does the patient have immunosuppression : ", 0.00, 1.00, value=0., step = 0.05, format="%f")
        hepatic_failure = st.slider("Does the patient has hepatic_failure: ", 0.00, 1.00, value=0., step = 0.05, format="%f")
        diabetes_mellitus = st.slider("Does the patient have diabetes_mellitus: ",0.00, 1.00, value=0., step = 0.05, format="%f")
        cirrhosis = st.slider("Does the patient have cirrhosis : ",0.00, 1.00, value=0.,step = 0.05, format="%f")
        aids = st.slider("Does the patient have aids : ", 0.00, 1.00, value=0.,step = 0.05, format="%f")    
        submit = st.form_submit_button("Predict")


    if submit:       
        apache_2_bodysystem = ordinal_encoder(apache_2_bodysystem, options_apache_2_bodysystem)
        #apache_3j_bodysystem = labelencoder(apache_3j_bodysystem, options_apache_3j_bodysystem)        


        data = np.array([apache_2_bodysystem,solid_tumor_with_metastasis,lymphoma,leukemia,
       immunosuppression,hepatic_failure,diabetes_mellitus,cirrhosis,aids]).reshape(1,-1)        
        print(data)
        
        pred = get_prediction(data=data, model=model)
        
        
        if pred[0] < 0.5:
            st.write(f"The patient belongs to class :  {pred[0]} and will not survive")
        else :
            st.write(f"The patient belongs to class :  {pred[0]} and will survive")


if __name__ == '__main__':
    main()
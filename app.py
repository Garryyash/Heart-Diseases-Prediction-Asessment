# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:26:15 2022

@author: caron
"""

import pickle
import os
import streamlit as st
import numpy as np

MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

    
#%% Streamlit
with st.form("Patient's Form"):
    st.title("Heart Attack Prediction")
    Age = st.number_input('Age')
    Sex = st.number_input('Sex')
    cp = st.number_input('cp')
    trtbps = st.number_input('trtbps')
    chol = st.number_input('chol')
    fbs = st.number_input('fbs')
    rest_ecg = st.number_input('rest_ecg')
    thalachh = st.number_input('thalach')
    exng = st.number_input('exng')
    oldpeak = st.number_input('oldpeak')
    slp = st.number_input('slp')
    caa = st.number_input('caa')
    thall = st.number_input('thall')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        temp = np.expand_dims([Age,Sex,cp,trtbps,chol,fbs,rest_ecg,thalachh,exng,oldpeak,
                 slp,caa,thall], axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Less chance of heart attack',
                        1:'More chance of heart attack'}
        
        if outcome == 1:
            st.snow()
            st.markdown('**High possibility** to get a heart attack!')
            st.write("Please change your lifestyle, make your heart healthy and young!")
            st.image("https://www.padbergcorrigan.com/wp-content/uploads/2017/02/Preventing-Heart-Disease-Infographic.png")
        else:
            st.balloons()
            st.write("Voila, you have a young heart age. Please keep your healthy lifestyle!")
            st.image("https://www.osfhealthcare.org/blog/wp-content/uploads/2020/08/Healthy-Aging-Infographic-768x960.png")





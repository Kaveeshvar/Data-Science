# %%
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from sklearn.preprocessing import StandardScaler

# %%
model = pickle.load(open('log16.pkl', 'rb'))

# %%
scale = pickle.load(open('std_sca.pkl', 'rb'))

# %%
st.title('Model Deployment using Logistic Regression')

# %%
def user_input_parameters():
    Gen = st.sidebar.radio("Select your gender",('Male','Female'))
    Ins = st.sidebar.selectbox("Do you have insurance? Yes-1' 'No-0",[0,1])
    Seat = st.sidebar.selectbox("Do you have Seatbelt? Yes-1 No-0",[0,1])
    Age = st.sidebar.slider('Age',1,100)
    loss = st.sidebar.number_input('Loss')
    data={'CLMSEX':Gen,
          'CLMINSUR':Ins,
          'SEATBELT':Seat,
          'CLMAGE':Age,
          'LOSS':loss}
    features=pd.DataFrame(data,index=[0])
    features['CLMSEX']=features['CLMSEX'].map({'Male':1,'Female':0})
    features[['CLMAGE','LOSS']]=scale.transform(features[['CLMAGE','LOSS']]) # This is done because we have to scale the features as we have done in training phase
    return features

# %%
df = user_input_parameters()
pred=model.predict(df)
pred_prob = model.predict_proba(df)
button=st.button('Predict')
if button is True:
    st.subheader('Prediction Result')
    st.write('Eligible for claim settlement' if pred_prob[0][1]>=0.5 else 'Not eligible for claim settlement')
    st.subheader('Prediction Probability')
    st.write(pred_prob)

# %%
df

# %%
 



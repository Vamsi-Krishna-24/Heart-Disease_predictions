#!/usr/bin/env python
# coding: utf-8

# In[171]:


import os 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st


# In[172]:


data = pd.read_csv('/Users/surisettivamsikrishna/Downloads/untitled folder/Projects /Heart prediction/framingham.csv')
data.isnull().any(axis=1)
data =data.dropna()


# In[173]:


del data['education']
del data['currentSmoker']
del data['prevalentHyp']
del data['sysBP']


# In[175]:


model = LogisticRegression(max_iter=1000)


# In[176]:


train = data.iloc[656:]
test = data.iloc[:656]


# In[177]:


predictors = ["male","age","cigsPerDay","BPMeds","prevalentStroke","diabetes","totChol","diaBP","BMI","heartRate","glucose"]


# In[178]:


model.fit(train[predictors],train["TenYearCHD"])


# In[179]:


preds = model.predict(test[predictors])


# In[180]:


test['predictions']=preds


# In[181]:


A=accuracy_score(test["TenYearCHD"],test["predictions"])
A


# In[182]:


st.title("Heart Disease Predictors")
st.write("Enter your information below to predict the likelihood of heart disease.")


# In[183]:


male = st.slider("Are you male? Select 1 for Yes, 0 for No:", 0, 1, 0)
age = st.slider("Enter your age:", 20, 80, 40)
cigs_per_day = st.slider("Enter the number of cigarettes per day:", 0, 40, 10)
bp_meds = st.slider("Are you on blood pressure medication? Select 1 for Yes, 0 for No:", 0, 1, 0)
prevalent_stroke = st.slider("Have you had a prevalent stroke? Select 1 for Yes, 0 for No:", 0, 1, 0)
diabetes = st.slider("Do you have diabetes? Select 1 for Yes, 0 for No:", 0, 1, 0)
tot_chol = st.slider("Enter your total cholesterol level:", 100, 300, 200)
dia_bp = st.slider("Enter your diastolic blood pressure:", 50, 120, 80)
bmi = st.slider("Enter your BMI:", 15.0, 40.0, 25.0)
heart_rate = st.slider("Enter your heart rate:", 50, 100, 72)
glucose = st.slider("Enter your glucose level:", 60, 200, 80)


# In[184]:


# Create a DataFrame with user input
new_data = pd.DataFrame({
    "male": [male],
    "age": [age],
    "cigsPerDay": [cigs_per_day],
    "BPMeds": [bp_meds],
    "prevalentStroke": [prevalent_stroke],
    "diabetes": [diabetes],
    "totChol": [tot_chol],
    "diaBP": [dia_bp],
    "BMI": [bmi],
    "heartRate": [heart_rate],
    "glucose": [glucose]
})


# In[185]:


prediction = model.predict(new_data[predictors])[0]

# Display prediction with custom styling
st.markdown(
    f"""
    <div style="font-size: 24px; padding: 20px; background-color: #f4f4f4; border-radius: 10px;">
        Predicted Outcome: <span style="color: {'green' if prediction == 0 else 'red'}; font-weight: bold;">{'Negative' if prediction == 0 else 'Positive'}</span>
    </div>
    """,
    unsafe_allow_html=True
)

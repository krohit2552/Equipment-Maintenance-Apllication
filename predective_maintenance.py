import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv('created_iot_data.csv')
df=data
model=joblib.load('predictive_maintenance_model.pkl')

# Data Prepration
data['maintenance_required'] = np.where(
    (data['temperature'] > 80) | (data['vibration'] > 5) | (data['pressure'] > 500), 1, 0)

data = data.drop(columns=['timestamp'])
X = data.drop(columns=['maintenance_required'])
y = data['maintenance_required']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model training
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# #Testing model 
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]

# Evaluating
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Classification Model Accuracy: {accuracy}")
# print(classification_report(y_test, y_pred))

def predict_maintenance(features):
    pred = model.predict([features])
    return 'Needs Maintenance' if pred[0] == 1 else 'Normal'



# Streamlit code
st.set_page_config(
    
    initial_sidebar_state="collapsed"
  
)
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Input Data", "Visualizations"],
        icons=[ "table", "input-cursor", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("Welcome to 'the Predictive Equipment Maintenance Application using Iot Data' ")
    st.markdown("""
    This application provides predictive maintenance insights for industrial Equipment.""")
    st.title("📂 Data")
    st.write(data.head(15))

elif selected == "Input Data":
    st.title("🔧 Input Features")
    
    rand_temprature = np.random.uniform(data['temperature'].min(), data['temperature'].max())
    rand_vibration = np.random.uniform(data['vibration'].min(), data['vibration'].max())
    rand_pressure = np.random.uniform(data['pressure'].min(), data['pressure'].max())

    Temperature=st.text_input('Enter Temperature','50')
    Vibratio=st.text_input('Enter Vibration','3')
    Pressure=st.text_input('Enter Pressure','400')

    
    if st.button('Submit'):
         st.title(" Prediction Results")
         input_features=[Temperature,Vibratio,Pressure]
         prediction = predict_maintenance(input_features)
         st.write(f"**Maintenance Status:** {prediction}")    


elif selected == "Visualizations":
    st.title("📊 Data Visualizations")


    st.subheader("Histogram of Sensors Readings")
    st.write('Histogram for Temprature')
    fig=plt.figure(figsize=(16,8))
    sns.histplot(df['temperature'], bins=30, kde=True).set_title('Temperature')
    st.pyplot(fig) 

    st.write('Histogram for Vibration')
    fig2=plt.figure(figsize=(16,8))
    sns.histplot(df['vibration'], bins=30, kde=True).set_title('Vibration')
    st.pyplot(fig2)

    st.write('Histogram for Pressure') 
    fig3=plt.figure(figsize=(16,8))
    sns.histplot(df['pressure'], bins=30, kde=True).set_title('Pressure')
    st.pyplot(fig3)

    
    st.title("📊 Sensors Data")
    fig4=plt.figure(figsize=(16, 10))
    sns.lineplot(df, x='timestamp', y='temperature', label='Temperature')
    sns.lineplot(df, x='timestamp', y='vibration', label='Vibration')
    sns.lineplot(df, x='timestamp', y='pressure', label='Pressure')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Readings')
    plt.legend()
    plt.title('Created IoT Sensor Data')
    st.pyplot(fig4)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    
    st.title("📊 Feature Importance")
    fig5=plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    st.pyplot(fig5)
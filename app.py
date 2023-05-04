import numpy as np
import pickle
import streamlit as st

# Loading the Model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

def Diabetes_Prediction(input_values):
    
    
    input_array = np.asanyarray(input_values)
    input_array_reshaped = input_array.reshape(1,-1)
    answer = loaded_model.predict(input_array_reshaped)  # It's a list and not an Integer
    if (answer[0] == 0):
        return 'The Person is Non-Diabetic'
    else:
        return 'The Person is Diabetic'
    
def main():
    
    st.title('Diabetes Prediction Web App')
    
    # Taking Inputs 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Levels')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('SkinThickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('Age of the Person')
    
    diagnosis = ''  # A Null String to store final Value
    
    if st.button('Test Result'):
        diagnosis = Diabetes_Prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()

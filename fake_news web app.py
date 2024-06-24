import numpy as np
import joblib
import streamlit as st
import sklearn
from scipy import sparse


# loading the saved model
loaded_model = joblib.load(open('logreg_model.pkl', 'rb'))
# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')


# creating a function for Prediction

def fake_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    proba = loaded_model.predict_proba(input_data_reshaped)

    if (prediction[0] == 0):
      return ['The news is fake', proba]
    else:
      return ['The news is not fake', proba]
  
    
  
def main():
    
    
    # giving a title
    st.title('Fake news Prediction Web App for USA\'s elections ')
    
    
    # getting the input data from the user
    
    
    title = st.text_input('Enter the title')
    author = st.text_input('Enter the author')
    text = st.text_input('Enter the text')

    final = title + ' ' + author+' '+text
    


    final = vectorizer.transform([final])

    
    # code for Prediction
    diagnosis = ''
    proba = ''
    
    # creating a button for Prediction
    
    if st.button('Fake news Test Result'):
        diagnosis = fake_prediction([final])[0]
        proba = fake_prediction([final])[1]
         
    st.success('Diagnostic: '+ diagnosis)
    st.success('Probability: '+ proba)

    
    
    
    
    
if __name__ == '__main__':
    main()
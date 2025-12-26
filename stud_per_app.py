import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Ashmit:ashmit1234@cluster0.entgrhi.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# Creating the database named student_performance_db
db = client['student_performance_db']
# Creating the collection named student_performance_predictions
collection = db['student_performance_predictions']
    


# Loading the pre-trained model and preprocessors
def load_model():
    with open('student_performance_model.pkl','rb') as file:
        model,scaler,le = pickle.load(file)
        return model,scaler,le
    
# Preprocessing input data
def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# Predicting student performance
def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.title('Student Performance Prediction')
    st.write('Enter the following details to predict student performance:')

    hours_studied = st.number_input('Hours Studied',min_value = 1, max_value= 10,value = 5)
    previous_scores = st.number_input('Previous Scores',min_value = 40, max_value= 100,value = 70)
    extracurricular_activities = st.selectbox('Extracurricular Activities',['Yes','No'])
    sleep_hours = st.number_input('Sleep Hours',min_value = 4, max_value= 9,value = 6)
    no_paper_solved = st.number_input('Number of Papers Solved',min_value = 10, max_value= 100,value = 20)

    if st.button('Predicted Performance'):
        user_data = {
            'Hours Studied': hours_studied,
            'Previous Scores': previous_scores,
            'Extracurricular Activities': extracurricular_activities,
            'Sleep Hours': sleep_hours,
            'Sample Question Papers Practiced': no_paper_solved
        }
        prediction = predict_data(user_data)
        print("The predicted performance is:", prediction,type(prediction))
        st.success(f'Predicted Performance : {prediction}')
        user_data['prediction'] = round(float(prediction[0]),2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)

if __name__ =='__main__':
    main()
import streamlit as st
from joblib import load


#load model from joblib file
model = load('./iris_model.pkl')

#create an web app
# create a title for your app
st.title('Iris Flower Prediction App')

# create streamlit widgets
#specy_names = [['setosa', 'versicolor', 'virginica']]

# create an input slider for each feature
sepal_length = st.slider('Sepal Length:', 4.3, 7.9, 5.4)    # (label, min, max, default)
sepal_width = st.slider('Sepal Width:', 2.0, 4.4, 3.4)   # (label, min, max, default)
petal_length = st.slider('Petal Length:', 1.0, 6.9, 1.3)  # (label, min, max, default)  
petal_width = st.slider('Petal Width:', 0.1, 2.5, 0.2)   # (label, min, max, default)

# create the feature vector
features = [[sepal_length, sepal_width, petal_length, petal_width]] # 2D array (matrix) with 1 row and 4 columns (1x4) 

# create a button to predict the specy
if st.button('Predict'):
    prediction = model.predict(features) # 2D array (matrix) with 1 row and 1 column (1x1)
    st.write(f'Your prediction is: {prediction[0]}') # 1D array (vector) with 1 element (1x1)
 
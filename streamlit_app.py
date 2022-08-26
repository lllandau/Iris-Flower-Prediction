from doctest import set_unittest_reportflags
from sklearn import svm, datasets
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def load_pickle(model_pickle_path):
    """
    Loading the saved pickle with model 
    """
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)


    return model

def make_predictions(df, model):
    prediction = model.predict(df)
    return prediction

def generate_predictions(df):
    model_pickle_path = "iris_prediction_model.pkl"

    model = load_pickle(model_pickle_path)
    prediction = make_predictions(df, model)

    return int(prediction[0])

def point_actualization(input):
  return 



if __name__ == "__main__":

  st.title("_Iris Flower Prediction_")

  iris = datasets.load_iris(as_frame=True)

  #iris_df = pd.DataFrame(iris)
  #st.write(iris_df)
  #st.write(type(iris_df))

  st.write("## Here we show the iris data in a table:")

  st.dataframe(iris.frame)

  #iris_df = datasets.load_iris(return_X_y=True)
  #iris_df = pd.DataFrame(iris_df)
  #st.write(iris.frame.columns)
 

    
  st.write("### Select params for prediction: ")

  sepal_length = st.slider("What is the length of the sepal?:",
                       min_value=3.0, max_value=9.2, value=5.8)

  sepal_width = st.slider("What is the width of the sepal?:",
                       min_value=1.5, max_value=5.5, value=2.0)
  petal_length = st.slider("What is the length of the sepal?:",
                       min_value=1.0, max_value=3.8, value=7.0)
  petal_width = st.slider("What is the length of the sepal?:",
                       min_value=0.1, max_value=2.8, value=1.2)


  df = px.data.iris()

  figures = [px.scatter(df, x="sepal_length", y="sepal_width", color="species"),px.scatter(df, x="petal_length", y="petal_width", color="species")]
  fig = make_subplots(rows=1, cols=2)
  for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.update_layout(showlegend=False)
        fig.append_trace(figure["data"][trace], row=1, col=i+1) 
    fig.update_layout(showlegend=True)



  reference_point_sepal = go.Scatter(x=[sepal_length],
                            y=[sepal_width],marker_size=10, marker_color="black", 
                            showlegend=False)

  reference_point_petal = go.Scatter(x=[petal_length],
                            y=[petal_width],marker_size=10, marker_color="black", 
                            showlegend=False)

  fig.add_trace(reference_point_sepal, row=1, col=1)
  fig.add_trace(reference_point_petal, row=1, col=2)
  st.write(fig)


input_dict = {"sepal length (cm)": sepal_length,
              "sepal width (cm)": sepal_width, 
              "petal length (cm)": petal_length, 
              "petal width (cm)": petal_width}


#st.write([sepal_length, sepal_width, petal_length, petal_width])
st.write(input_dict)

input_data = pd.DataFrame([input_dict])
#st.write(input_data)

#st.write(generate_predictions(input_data))



if st.button("Predict Flower"):
  pred = generate_predictions(input_data)
  if pred == 0: 
    st.success("Flower is setosa!")
  elif pred == 1: 
    st.success("Flower is versicolor!")
  else: 
    st.success("Flower is virginica!")




from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling as yp
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder

import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Luke's Auto ML App")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = yp.ProfileReport(df)
    st_profile_report(profile_df)


if choice == "Modelling": 
    chosen_target = st.selectbox('Choose Target Column', df.columns)
    df = df.dropna(subset=[chosen_target])
    # df[chosen_target] = pd.to_numeric(df[chosen_target], errors='coerce')

    # Iterate through columns and handle data types
    for col in df.columns:
        # If boolean (two unique values), convert to integer
        if len(df[col].unique()) == 2:
            df[col] = df[col].astype(int)
        # If object, convert to category
        elif df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        # If numeric with NaN, ensure float type
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
            df[col] = df[col].astype(float)

    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
import streamlit as st
import pandas as pd

df = pd.read_csv('laptop_details.csv')
st.header("Laptop Data Given:")
st.dataframe(df)

df1 = pd.read_csv('laptop.csv')
df1 = df1.drop('Unnamed: 0',axis=1)
st.header('After Feature Engineering')
st.dataframe(df1)
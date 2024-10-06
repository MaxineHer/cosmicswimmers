import streamlit as st
import pandas as pd
import plotly.express as px

st.write("COSMIC SWIMMERS WEATHER DASHBOARD")

df = pd.read_csv("avgs.csv")
choices = ["temperature", "precipitation","windspeed"]
cat = st.selectbox("Choose Measure: ", choices, index=0)

if (cat == "temperature"):
    fig = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="temperature", title="Temperatures")
if (cat == "precipitation"):
    fig = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="precipitation", title="Precipitations")
if (cat == "windspeed"):
    fig = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="windspeed", title="Wind Speeds")

st.plotly_chart(fig, use_container_width=True)
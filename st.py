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
# newwwwie
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
# Load the data

df = pd.read_csv(
    "GlobalLandTemperaturesByCountry.csv"
)
# Convert 'dt' to datetime

df["dt"] = pd.to_datetime(df["dt"])
# Filter data for the last 50 years

current_year = datetime.now().year
df_recent = df[df["dt"].dt.year >= current_year - 50]
# Calculate average temperature for each country

df_avg = df_recent.groupby("Country")["AverageTemperature"].mean().reset_index()
# Create a choropleth map

fig_map = px.choropleth(
    df_avg,
    locations="Country",
    locationmode="country names",
    color="AverageTemperature",
    hover_name="Country",
    color_continuous_scale="RdYlBu_r",
    title="Average Temperature by Country (Last 50 Years)",
)
fig_map.update_layout(
    geo=dict(
        showframe=False, showcoastlines=True, projection_type="natural earth"
    )
)

# Create a heatmap for temperature trends

df_pivot = df_recent.pivot(
    index="dt", columns="Country", values="AverageTemperature"
)

df_pivot = df_pivot.resample("Y").mean()


fig_heatmap = px.imshow(
    df_pivot.T,
    x=df_pivot.index.year,
    y=df_pivot.columns,
    color_continuous_scale="RdYlBu_r",
    title="Temperature Trends by Country (Heatmap)",
)

fig_heatmap.update_layout(
    xaxis_title="Year",
    yaxis_title="Country",
    coloraxis_colorbar=dict(title="Temperature (°C)"),
)

# Predict temperatures until November end
def predict_temperature(country):
    country_data = df[df["Country"] == country].sort_values("dt")
    country_data = country_data.dropna(subset=["AverageTemperature"])
    X = (country_data["dt"] - country_data["dt"].min()).dt.days.values.reshape(
        -1, 1
    )
    y = country_data["AverageTemperature"].values
    model = LinearRegression()
    model.fit(X, y)
    last_date = country_data["dt"].max()
    future_dates = pd.date_range(start=last_date, end="2023-11-30", freq="D")
    future_X = (future_dates - country_data["dt"].min()).days.values.reshape(
        -1, 1
    )
    predictions = model.predict(future_X)
    return future_dates, predictions
# Create a line plot with predictions for a few countries

countries_to_predict = ["United States", "China", "India", "Brazil", "Russia"]
fig_prediction = make_subplots(rows=1, cols=1, shared_xaxes=True)
for country in countries_to_predict:

    country_data = df[df["Country"] == country].sort_values("dt")

    fig_prediction.add_trace(
        go.Scatter(
            x=country_data["dt"],
            y=country_data["AverageTemperature"],
            name=f"{country} (Historical)",
            mode="lines",
        )
    )
    future_dates, predictions = predict_temperature(country)

    fig_prediction.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            name=f"{country} (Predicted)",
            mode="lines",
            line=dict(dash="dash"),
        )
    )
fig_prediction.update_layout(
    title="Temperature Predictions until November 2023",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    legend_title="Country",
)
# Display the figures

st.plotly_chart(fig_map, use_container_width=True)
st.plotly_chart(fig_heatmap, use_container_width=True)
st.plotly_chart(fig_prediction, use_container_width=True)
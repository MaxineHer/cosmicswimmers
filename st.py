import streamlit as st
import pandas as pd
import plotly.express as px

st.write("COSMIC SWIMMERS WEATHER DASHBOARD")

df = pd.read_csv("avgs.csv")
choices = ["temperature", "precipitation","windspeed"]
cat = st.selectbox("Choose Measure: ", choices, index=0)

if (cat == "temperature"):
    fig1 = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="temperature", title="Temperatures")
if (cat == "precipitation"):
    fig1 = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="precipitation", title="Precipitations")
if (cat == "windspeed"):
    fig1 = px.choropleth(data_frame=df, locationmode="country names", scope="world", locations="Country", color="windspeed", title="Wind Speeds")
st.plotly_chart(fig1)
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

st.plotly_chart(fig_map )
st.plotly_chart(fig_heatmap)
st.plotly_chart(fig_prediction)

file_path = (
    "GlobalLandTemperaturesByCountry.csv"
)

climate_data = pd.read_csv(file_path)


# Preview the first few rows of the data

print(climate_data.head())

# Convert the 'dt' column to datetime format

climate_data["dt"] = pd.to_datetime(climate_data["dt"])


# Filter data to include only November records

climate_data_november = climate_data[climate_data["dt"].dt.month == 11]


# Optionally, focus on a specific country (e.g., "United States")

climate_data_november_us = climate_data_november[
    climate_data_november["Country"] == "United States"
]


# Preview the filtered data

print(climate_data_november_us.head())


# basically an interactive line graph for November temperatures in the US

fig2 = px.line(
    climate_data_november_us,
    x="dt",
    y="AverageTemperature",
    title="November Temperature Trends in the US",
    labels={"AverageTemperature": "Average Temperature (°C)", "dt": "Date"},
    hover_data=["Country"],
)


# Customizeee the plot

fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Temperature (°C)",
    title_font_size=20,
    template="plotly_dark",
)


# Show the plot


st.plotly_chart(fig2)

# Load the dataset (replace with your file path)

df = pd.read_csv(
    "GlobalLandTemperaturesByCountry.csv"
)


# Convert 'dt' column to datetime

df["dt"] = pd.to_datetime(df["dt"])


# Filter the data for the month of November across all years

df_november = df[df["dt"].dt.month == 11]


df_november_avg = (
    df_november.groupby([df_november["dt"].dt.year, "Country"])[
        "AverageTemperature"
    ]
    .mean()
    .reset_index()
)


# Rename columns for clarity

df_november_avg.columns = ["Year", "Country", "AvgTemperature"]


# Preview the data

df_november_avg.head()

fig = px.line(
    df_november_avg,
    x="Year",
    y="AvgTemperature",
    color="Country",
    title="Average November Temperature Trends by Country",
    labels={"Year": "Year", "AvgTemperature": "Average Temperature (°C)"},
    template="plotly_dark",  # Attractive dark theme
)

fig.update_layout(
    font=dict(size=16),
    hovermode="x unified",  # Combine hover labels for easy reading
    xaxis=dict(title="Year", tickformat="%Y", showgrid=True),
    yaxis=dict(title="Avg Temperature (°C)", showgrid=True),
    legend=dict(
        title="Country",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)


# Display the plot

st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="ClimateScope", layout="wide")

USERS_FILE = "users.csv"
DATA_FILE = "data/climate_data.csv"

# ---------------- AUTH UTILS ---------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["username", "password"])
    return pd.read_csv(USERS_FILE)

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    users.loc[len(users)] = [username, hash_password(password)]
    users.to_csv(USERS_FILE, index=False)
    return True

def authenticate(username, password):
    users = load_users()
    return not users[
        (users["username"] == username) &
        (users["password"] == hash_password(password))
    ].empty

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ---------------- #
def login_page():
    st.title("🌍 ClimateScope – Visualizing Global Weather Trends")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")

# ---------------- DATA CLEANING ---------------- #
def load_and_clean_data():
    df = pd.read_csv(DATA_FILE)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    st.write("✅ Columns after cleaning:", df.columns.tolist())

    # Ensure datetime column exists
    if "datetime" not in df.columns:
        st.error("❌ 'datetime' column not found. Check dataset header.")
        st.stop()

    # Remove empty rows
    df = df.dropna(how="all")

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Extract year and month
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    # Convert numeric columns
    numeric_cols = [
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "wind_kph"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df

# ---------------- DASHBOARD ---------------- #
def dashboard():
    st.sidebar.title("🌡 ClimateScope")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.header("📊 Global Temperature Analysis")

    df = load_and_clean_data()

    # -------- DATA PREVIEW -------- #
    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(df.head(50), use_container_width=True)

    # -------- SUMMARY -------- #
    st.subheader("📈 Statistical Summary")
    st.write(df.describe())

    # -------- TEMPERATURE TREND -------- #
    st.subheader("🌡 Average Temperature Trend (Year-wise)")

    yearly_temp = (
        df.groupby(["year", "country"])["temperature_celsius"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        yearly_temp,
        x="year",
        y="temperature_celsius",
        color="country",
        title="Year-wise Average Temperature by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------- EXTREME EVENTS -------- #
    st.subheader("🔥 Extreme Temperature Events")

    threshold = df["temperature_celsius"].mean() + df["temperature_celsius"].std()

    extreme_df = df[df["temperature_celsius"] > threshold]

    st.write(f"Threshold used: **{threshold:.2f} °C**")

    st.dataframe(
        extreme_df[
            ["country", "datetime", "temperature_celsius", "humidity", "wind_kph"]
        ],
        use_container_width=True
    )
# Milestone 4 update
# Milestone 4 update - interactive dashboard
# ---------------- MAIN ---------------- #
if st.session_state.logged_in:
    dashboard()
else:
    login_page()
# Added monthly temperature trend and cleaned datetime handling

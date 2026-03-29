import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="ClimateScope", layout="wide")

USERS_FILE = "users.csv"
DATA_FILE = "data/climate_data.csv"

# ---------------- CACHE ---------------- #
@st.cache_data
def load_and_clean_data():
    if not os.path.exists(DATA_FILE):
        st.error("Climate dataset not found.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    numeric_cols = ["temperature_celsius", "precip_mm", "humidity", "wind_kph"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()

# ---------------- AUTH ---------------- #
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
    st.title("🌍 ClimateScope – Interactive Climate Dashboard")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("Registered successfully!")
            else:
                st.error("User already exists")

# ---------------- DASHBOARD ---------------- #
def dashboard():

    st.sidebar.title("🌡 ClimateScope")

    menu = st.sidebar.radio(
        "Navigation",
        ["Overview", "Interactive Charts", "Choropleth Map", "Advanced Insights"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    df = load_and_clean_data()

    # ---------------- FILTERS ---------------- #
    st.sidebar.subheader("🎛 Filters")

    countries = st.sidebar.multiselect(
        "🌍 Countries",
        df["country"].unique(),
        default=df["country"].unique()   # FIXED
    )

    year_range = st.sidebar.slider(
        "📅 Year Range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max()))
    )

    temp_range = st.sidebar.slider(
        "🌡 Temperature Range",
        float(df["temperature_celsius"].min()),
        float(df["temperature_celsius"].max()),
        (float(df["temperature_celsius"].min()), float(df["temperature_celsius"].max()))
    )

    df = df[
        (df["country"].isin(countries)) &
        (df["year"].between(*year_range)) &
        (df["temperature_celsius"].between(*temp_range))
    ]

    # ---------------- EMPTY CHECK ---------------- #
    if df.empty:
        st.warning("⚠️ No data available for selected filters. Please adjust filters.")
        return

    # ---------------- METRICS ---------------- #
    st.title("📊 Climate Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌡 Avg Temp", round(df["temperature_celsius"].mean(), 2))
    col2.metric("💧 Humidity", round(df["humidity"].mean(), 2))
    col3.metric("🌧 Rainfall", round(df["precip_mm"].mean(), 2))
    col4.metric("💨 Wind", round(df["wind_kph"].mean(), 2))

    # ---------------- OVERVIEW ---------------- #
    if menu == "Overview":

        st.subheader("📌 Quick Insights")

        with st.expander("🔍 Show Insights"):
            grouped = df.groupby("country")["temperature_celsius"].mean()

            if not grouped.empty:
                st.write("🔥 Hottest Country:", grouped.idxmax())
                st.write("❄ Coldest Country:", grouped.idxmin())
            else:
                st.write("No data available")

        st.dataframe(df, use_container_width=True)

    # ---------------- INTERACTIVE CHARTS ---------------- #
    elif menu == "Interactive Charts":

        st.subheader("📊 Build Your Own Chart")

        chart_type = st.selectbox("Chart Type", ["Line", "Scatter", "Histogram", "Box"])
        x_axis = st.selectbox("X-axis", df.columns)
        y_axis = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns)
        color = st.selectbox("Color By", ["country", "year", "month"])

        if chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=color)

        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, size="humidity")

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color)

        elif chart_type == "Box":
            fig = px.box(df, x="country", y=y_axis, color="country")

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- MAP ---------------- #
    elif menu == "Choropleth Map":

        st.subheader("🌍 Global Climate Map")

        metric = st.selectbox(
            "Select Metric",
            ["temperature_celsius", "humidity", "precip_mm", "wind_kph"]
        )

        country_avg = df.groupby("country")[metric].mean().reset_index()

        fig = px.choropleth(
            country_avg,
            locations="country",
            locationmode="country names",
            color=metric,
            hover_name="country",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ADVANCED ---------------- #
    elif menu == "Advanced Insights":

        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔥 Extremes", "🧠 Similarity"])

        # Trends
        with tab1:
            yearly = df.groupby(["year", "country"])["temperature_celsius"].mean().reset_index()
            st.plotly_chart(px.line(yearly, x="year", y="temperature_celsius", color="country"))

        # Extremes
        with tab2:
            threshold = st.slider(
                "Extreme Temperature Threshold",
                float(df["temperature_celsius"].min()),
                float(df["temperature_celsius"].max()),
                float(df["temperature_celsius"].mean())
            )

            extreme_df = df[df["temperature_celsius"] > threshold]

            st.write("🔥 Extreme Events:", len(extreme_df))
            st.plotly_chart(px.bar(extreme_df, x="country", y="temperature_celsius", color="country"))

        # Similarity
        with tab3:
            country_features = df.groupby("country").mean(numeric_only=True)

            countries_list = country_features.index.tolist()

            if len(countries_list) >= 2:
                c1 = st.selectbox("Country 1", countries_list)
                c2 = st.selectbox("Country 2", countries_list, index=1)

                scaler = StandardScaler()
                scaled = scaler.fit_transform(country_features)

                sim = cosine_similarity(
                    [scaled[countries_list.index(c1)]],
                    [scaled[countries_list.index(c2)]]
                )[0][0]

                st.success(f"🧠 Similarity Score: {round(sim*100,2)}%")
            else:
                st.warning("Not enough data for similarity comparison")

    # ---------------- DOWNLOAD ---------------- #
    st.sidebar.download_button(
        "📥 Download Filtered Data",
        df.to_csv(index=False),
        file_name="filtered_climate_data.csv"
    )

# ---------------- MAIN ---------------- #
if st.session_state.logged_in:
    dashboard()
else:
    login_page()
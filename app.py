
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================================
# CONFIGURATION
# =========================================

st.set_page_config(
    page_title="WattWise | Energy Intelligence",
    page_icon="⚡",
    layout="wide"
)

# =========================================
# LOAD MODEL AND PROCESSED DATA
# =========================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "random_forest_model.pkl.gz"
    features_path = BASE_DIR / "models" / "model_features.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure models/random_forest_model.pkl is committed to GitHub."
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Make sure models/model_features.pkl is committed to GitHub."
        )

    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
    return model, model_features


@st.cache_data(show_spinner="Loading processed household electricity data...")
def load_energy_data():
    data_path = BASE_DIR / "dataset" / "processed_hourly_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. "
            "Make sure dataset/processed_hourly_data.csv is committed to GitHub."
        )

    # This CSV already contains hourly energy values and engineered features.
    # Do not repeat raw-data cleaning, resampling, or feature engineering here.
    ml_df = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True
    )
    ml_df.index = pd.to_datetime(ml_df.index)

    ml_df["hour_sin"] = np.sin(2 * np.pi * ml_df["hour"] / 24)
    ml_df["hour_cos"] = np.cos(2 * np.pi * ml_df["hour"] / 24)
    
    ml_df["day_sin"] = np.sin(2 * np.pi * ml_df["day_of_week"] / 7)
    ml_df["day_cos"] = np.cos(2 * np.pi * ml_df["day_of_week"] / 7)
    
    ml_df["month_sin"] = np.sin(2 * np.pi * (ml_df["month"] - 1) / 12)
    ml_df["month_cos"] = np.cos(2 * np.pi * (ml_df["month"] - 1) / 12)
    
    required_columns = [
        "energy_kwh", "hour", "day_of_week", "month", "is_weekend",
        "previous_hour", "previous_day", "rolling_mean_24",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "month_sin", "month_cos"
    ]
    missing_columns = [col for col in required_columns if col not in ml_df.columns]
    if missing_columns:
        raise ValueError(
            "The processed dataset is missing required columns: "
            + ", ".join(missing_columns)
            + ". Re-create processed_hourly_data.csv from the final ml_df in your notebook."
        )

    # The exported file is the cleaned feature dataset (ml_df), so it is also
    # used as the available hourly history for the dashboard charts.
    hourly_df = ml_df.copy()
    return hourly_df, ml_df


# Load the saved model, feature names, and processed data.
model, model_features = load_model()
hourly_df, ml_df = load_energy_data()

# Recreate the same chronological 80/20 test split used during model training.
missing_model_features = [col for col in model_features if col not in ml_df.columns]
if missing_model_features:
    raise ValueError(
        "The processed dataset does not contain model features: "
        + ", ".join(missing_model_features)
    )

X_all = ml_df[model_features]
y_all = ml_df["energy_kwh"]
split_index = int(len(ml_df) * 0.8)
X_test_graph = X_all.iloc[split_index:]
y_test_graph = y_all.iloc[split_index:]

# Predictions for the held-out test period and their residuals.
y_pred_graph = model.predict(X_test_graph)
residuals = y_test_graph.to_numpy() - y_pred_graph

# =========================================
# CUSTOM DESIGN
# =========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080F1E;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101C32, #11152B);
    border-right: 1px solid #24334D;
}

[data-testid="stSidebar"] * {
    color: #DCE8F7;
}

h1 {
    color: #F8FAFC !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #E2E8F0 !important;
}

p, label {
    color: #A8B8CC;
}

.hero {
    background: linear-gradient(115deg, #123B55, #26316B, #43256C);
    padding: 35px;
    border-radius: 22px;
    border: 1px solid #405D91;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px !important;
    color: white !important;
}

.hero p {
    color: #D6E5FA;
    font-size: 16px;
}

.kpi {
    background: linear-gradient(145deg, #15243B, #182D49);
    border: 1px solid #2C4567;
    padding: 23px;
    border-radius: 18px;
    min-height: 130px;
}

.kpi-label {
    color: #A9BCD3;
    font-size: 13px;
}

.kpi-value {
    color: #67E8F9;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.kpi-sub {
    color: #93A8C2;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #121E32;
    border: 1px solid #293D59;
    padding: 23px;
    border-radius: 18px;
    margin-bottom: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #0891B2, #6366F1);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #67E8F9;
}

div[data-testid="stMetric"] {
    background: #15243B;
    border: 1px solid #2C4567;
    padding: 18px;
    border-radius: 15px;
}

div[data-testid="stMetricValue"] {
    color: #67E8F9;
}

hr {
    border-color: #273B57;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HELPER FOR CHARTS
# =========================================

def chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE8F7"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(
        gridcolor="#25364F",
        zerolinecolor="#25364F"
    )
    fig.update_yaxes(
        gridcolor="#25364F",
        zerolinecolor="#25364F"
    )
    return fig

# =========================================
# SIDEBAR
# =========================================

st.sidebar.markdown("# ⚡ WattWise")
st.sidebar.caption("ENERGY INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Overview",
        "Predict Energy",
        "Analytics",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### MODEL STATUS")
st.sidebar.success("Random Forest model loaded")

st.sidebar.markdown("---")
st.sidebar.caption("WattWise • Machine Learning Project")

# =========================================
# OVERVIEW
# =========================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <p>WELCOME TO YOUR ENERGY DASHBOARD</p>
        <h1>Smarter Energy.<br>Better Decisions. ⚡</h1>
        <p>
        Understand household electricity patterns and
        forecast consumption using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Dashboard Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="kpi">
            <div class="kpi-label">PREDICTION ENGINE</div>
            <div class="kpi-value">Active</div>
            <div class="kpi-sub">Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="kpi">
            <div class="kpi-label">TRAINING DATA</div>
            <div class="kpi-value">33,791</div>
            <div class="kpi-sub">Usable hourly records</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi">
            <div class="kpi-label">MODEL R² SCORE</div>
            <div class="kpi-value">62.40%</div>
            <div class="kpi-sub">Original Random Forest evaluation</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("""
        <div class="panel">
        <h3>⚡ What is WattWise?</h3>
        <p>
        WattWise is a household electricity forecasting
        project that uses historical energy consumption
        patterns to estimate future usage.
        </p>
        <p>
        It combines data preprocessing, feature engineering,
        machine learning and interactive visualization.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
        <h3>🚀 Platform Features</h3>
        <p>✓ Hourly energy prediction</p>
        <p>✓ Interactive analytics</p>
        <p>✓ Model performance comparison</p>
        <p>✓ Machine learning insights</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Explore WattWise")

    a, b, c = st.columns(3)

    with a:
        st.info("⚡ Predict Energy\n\nEstimate electricity consumption.")

    with b:
        st.info("📊 Analytics\n\nExplore your prediction results.")

    with c:
        st.info("🤖 Model Insights\n\nUnderstand model performance.")

# =========================================
# PREDICTION
# =========================================

elif page == "Predict Energy":

    st.title("⚡ Energy Prediction")
    st.caption("Enter historical consumption details to estimate the selected hour.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Household Consumption Inputs")

    a, b, c = st.columns(3)

    with a:
        previous_hour = st.number_input(
            "Previous hour (kWh)", 0.0, value=1.5
        )

    with b:
        previous_day = st.number_input(
            "Same hour yesterday (kWh)", 0.0, value=1.8
        )

    with c:
        rolling_mean_24 = st.number_input(
            "24-hour average (kWh)", 0.0, value=1.6
        )

    a, b, c = st.columns(3)

    with a:
        hour = st.slider("Hour", 0, 23, 18)

    with b:
        day = st.selectbox(
            "Day",
            range(7),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )

    with c:
        month = st.selectbox(
            "Month",
            range(1, 13),
            format_func=lambda x: [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ][x-1]
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Generate Energy Prediction", use_container_width=True):

        data = pd.DataFrame([{
            "is_weekend": int(day >= 5),
            "previous_hour": previous_hour,
            "previous_day": previous_day,
            "rolling_mean_24": rolling_mean_24,
            "hour_sin": np.sin(2*np.pi*hour/24),
            "hour_cos": np.cos(2*np.pi*hour/24),
            "day_sin": np.sin(2*np.pi*day/7),
            "day_cos": np.cos(2*np.pi*day/7),
            "month_sin": np.sin(2*np.pi*month/12),
            "month_cos": np.cos(2*np.pi*month/12)
        }])

        data = data[model_features]

        prediction = float(model.predict(data)[0])

        st.session_state["prediction"] = prediction
        st.session_state["previous_hour"] = previous_hour

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]
        previous = st.session_state["previous_hour"]

        st.success("Your prediction is ready!")

        c1, c2 = st.columns(2)

        c1.metric("Predicted Consumption", f"{prediction:.3f} kWh")
        c2.metric("Previous Hour", f"{previous:.3f} kWh")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Estimated Energy Consumption"},
            gauge={
                "axis": {"range": [0, max(5, prediction*1.2)]},
                "bar": {"color": "#22D3EE"},
                "steps": [
                    {"range": [0, max(5, prediction*1.2)*0.5],
                     "color": "#17334C"},
                    {"range": [max(5, prediction*1.2)*0.5,
                               max(5, prediction*1.2)],
                     "color": "#312E65"}
                ]
            }
        ))

        st.plotly_chart(chart_theme(gauge), use_container_width=True)

        difference = prediction - previous

        if difference > 0.1:
            st.warning("Consumption is predicted to increase.")
        elif difference < -0.1:
            st.success("Consumption is predicted to decrease.")
        else:
            st.info("Consumption is predicted to remain approximately similar.")

        st.caption("Prediction is an estimate, not a live electricity meter reading.")

# =========================================
# ANALYTICS
# =========================================


# =========================================
# ANALYTICS PAGE - ALL GRAPHS
# =========================================

elif page == "Analytics":

    st.title("📊 Energy Analytics")
    st.caption("Explore real household electricity consumption patterns.")

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Hourly Records",
        f"{len(hourly_df):,}"
    )

    c2.metric(
        "Valid Hourly Records",
        f"{hourly_df['energy_kwh'].notna().sum():,}"
    )

    c3.metric(
        "Average Consumption",
        f"{hourly_df['energy_kwh'].mean():.3f} kWh"
    )

    st.divider()

    # GRAPH 1: HOURLY CONSUMPTION TREND

    st.subheader("1. Hourly Electricity Consumption Trend")

    trend_data = hourly_df["energy_kwh"].dropna()

    fig1 = px.line(
        x=trend_data.index,
        y=trend_data.values,
        labels={
            "x": "Date",
            "y": "Energy Consumption (kWh)"
        },
        title="Household Electricity Consumption Over Time"
    )

    fig1.update_traces(
        line=dict(color="#22D3EE", width=1)
    )

    st.plotly_chart(chart_theme(fig1), use_container_width=True)

    # GRAPH 2: CONSUMPTION DISTRIBUTION

    st.subheader("2. Consumption Distribution")

    fig2 = px.histogram(
        hourly_df.dropna(subset=["energy_kwh"]),
        x="energy_kwh",
        nbins=50,
        title="Distribution of Hourly Electricity Consumption",
        labels={"energy_kwh": "Consumption (kWh)"},
        color_discrete_sequence=["#818CF8"]
    )

    st.plotly_chart(chart_theme(fig2), use_container_width=True)

    # GRAPH 3: HOURLY USAGE PATTERN

    st.subheader("3. Average Consumption by Hour")

    hourly_profile = (
        hourly_df.groupby("hour")["energy_kwh"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        hourly_profile,
        x="hour",
        y="energy_kwh",
        title="Average Electricity Consumption During the Day",
        labels={
            "hour": "Hour of Day",
            "energy_kwh": "Average Consumption (kWh)"
        },
        color="energy_kwh",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(chart_theme(fig3), use_container_width=True)

    # GRAPH 4: DAILY USAGE PATTERN

    st.subheader("4. Average Consumption by Day")

    day_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    daily_profile = (
        hourly_df.groupby("day_of_week")["energy_kwh"]
        .mean()
        .reset_index()
    )

    daily_profile["day"] = daily_profile["day_of_week"].map(
        lambda x: day_names[x]
    )

    fig4 = px.bar(
        daily_profile,
        x="day",
        y="energy_kwh",
        title="Average Electricity Consumption by Weekday",
        color="energy_kwh",
        color_continuous_scale="Purples"
    )

    st.plotly_chart(chart_theme(fig4), use_container_width=True)

    # GRAPH 5: MONTHLY USAGE PATTERN

    st.subheader("5. Monthly Consumption Pattern")

    monthly_profile = (
        hourly_df.groupby("month")["energy_kwh"]
        .mean()
        .reset_index()
    )

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_profile["month_name"] = monthly_profile["month"].map(
        lambda x: month_names[x - 1]
    )

    fig5 = px.line(
        monthly_profile,
        x="month_name",
        y="energy_kwh",
        markers=True,
        title="Average Electricity Consumption by Month"
    )

    fig5.update_traces(
        line=dict(color="#A78BFA", width=3),
        marker=dict(size=9)
    )

    st.plotly_chart(chart_theme(fig5), use_container_width=True)

    # GRAPH 6: ACTUAL VS PREDICTED

    st.subheader("6. Actual vs Predicted Consumption")

    comparison_count = min(300, len(y_test_graph))

    actual_predicted = pd.DataFrame({
        "Actual": y_test_graph.iloc[:comparison_count].values,
        "Predicted": y_pred_graph[:comparison_count]
    }, index=y_test_graph.index[:comparison_count])

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=actual_predicted.index,
        y=actual_predicted["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="#22D3EE")
    ))

    fig6.add_trace(go.Scatter(
        x=actual_predicted.index,
        y=actual_predicted["Predicted"],
        mode="lines",
        name="Predicted",
        line=dict(color="#A78BFA")
    ))

    fig6.update_layout(
        title="Actual vs Predicted Electricity Consumption",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)"
    )

    st.plotly_chart(chart_theme(fig6), use_container_width=True)

    # GRAPH 7: ACTUAL VS PREDICTED SCATTER

    st.subheader("7. Prediction Accuracy Scatter Plot")

    fig7 = px.scatter(
        x=y_test_graph.values,
        y=y_pred_graph,
        labels={
            "x": "Actual Consumption (kWh)",
            "y": "Predicted Consumption (kWh)"
        },
        title="Actual vs Predicted Values",
        opacity=0.4
    )

    low = min(y_test_graph.min(), np.min(y_pred_graph))
    high = max(y_test_graph.max(), np.max(y_pred_graph))

    fig7.add_shape(
        type="line",
        x0=low,
        y0=low,
        x1=high,
        y1=high,
        line=dict(color="#F472B6", dash="dash")
    )

    st.plotly_chart(chart_theme(fig7), use_container_width=True)

    # GRAPH 8: RESIDUAL DISTRIBUTION

    st.subheader("8. Prediction Error Distribution")

    residual_df = pd.DataFrame({
        "Residual": residuals
    })

    fig8 = px.histogram(
        residual_df,
        x="Residual",
        nbins=50,
        title="Distribution of Prediction Errors",
        color_discrete_sequence=["#F472B6"]
    )

    st.plotly_chart(chart_theme(fig8), use_container_width=True)

    # GRAPH 9: FEATURE IMPORTANCE

    st.subheader("9. Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig9 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Features Influencing Random Forest Predictions",
        color="Importance",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(chart_theme(fig9), use_container_width=True)

    # GRAPH 10: FEATURE CORRELATION

    st.subheader("10. Feature Correlation Heatmap")

    correlation_columns = [
        "energy_kwh",
        "previous_hour",
        "previous_day",
        "rolling_mean_24",
        "hour",
        "day_of_week",
        "month"
    ]

    correlation = ml_df[correlation_columns].corr()

    fig10 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Energy and Features"
    )

    st.plotly_chart(chart_theme(fig10), use_container_width=True)


# =========================================
# MODEL PERFORMANCE
# =========================================


# =========================================
# MODEL PERFORMANCE PAGE
# =========================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")
    st.caption("Compare all trained models using test-set evaluation metrics.")

    results = pd.read_csv("models/model_comparison.csv")

    # Metric comparison table
    st.subheader("Model Comparison")

    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    # Three metric charts
    for metric, description, better in [
        ("MAE", "Mean Absolute Error", "Lower is better"),
        ("RMSE", "Root Mean Squared Error", "Lower is better"),
        ("R2 Score", "R² Score", "Higher is better")
    ]:

        st.subheader(description)
        st.caption(better)

        fig = px.bar(
            results,
            x="Model",
            y=metric,
            color="Model",
            text_auto=".4f",
            title=f"{description} Comparison",
            color_discrete_sequence=["#22D3EE", "#A78BFA", "#F472B6", "#34D399"]
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    # Actual test performance
    st.divider()

    st.subheader("Test Set Prediction Quality")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    test_mae = mean_absolute_error(y_test_graph, y_pred_graph)
    test_rmse = np.sqrt(mean_squared_error(y_test_graph, y_pred_graph))
    test_r2 = r2_score(y_test_graph, y_pred_graph)

    a, b, c = st.columns(3)

    a.metric("Test MAE", f"{test_mae:.4f}")
    b.metric("Test RMSE", f"{test_rmse:.4f}")
    c.metric("Test R²", f"{test_r2:.4f}")

    st.info(
        "These scores are calculated from the loaded model and "
        "the recreated chronological test set."
    )

    # Feature importance
    st.subheader("Most Important Features")

    importance = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
        title="Random Forest Feature Importance"
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(chart_theme(fig), use_container_width=True)


# =========================================
# ABOUT
# =========================================

elif page == "About Project":

    st.title("About WattWise")

    st.markdown("""
    <div class="hero">
        <h1>Energy Intelligence ⚡</h1>
        <p>
        WattWise demonstrates how machine learning
        can help estimate household electricity consumption.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Technology Stack")

    a, b, c = st.columns(3)

    a.info("🐍 Python\n\nProgramming language")
    b.info("🌲 Scikit-learn\n\nMachine learning")
    c.info("📊 Streamlit\n\nInteractive dashboard")

    st.subheader("Project Workflow")

    st.write("""
    1. Data collection from UCI.
    2. Data cleaning and preprocessing.
    3. Exploratory data analysis.
    4. Hourly feature engineering.
    5. Machine learning model training.
    6. Model evaluation.
    7. Prediction system.
    8. Interactive dashboard.
    """)

    st.success("WattWise — Making Energy Forecasting Smarter.")

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# ------------------------------
# Streamlit page config & theme
# ------------------------------
st.set_page_config(page_title="Interactive Regression Lab", layout="wide", page_icon="📊")

# CSS for light pastel theme
st.markdown("""
<style>
body {background-color: #F5F9FF;}
.sidebar .sidebar-content {background-color: #E3F2FF;}
h1 {color:#004080;}
h2 {color:#0066CC;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Interactive Regression Learning Lab")
st.write("Interactive regression tool with light pastel theme and real-time updates.")

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("Controls")

# Regression type dropdown
model_type = st.sidebar.selectbox("Regression Model",
                                  ["OLS (Linear Regression)", "Ridge", "Lasso", "Elastic Net"])

# Lambda slider
lambda_val = st.sidebar.slider("λ (Regularization strength)", 0.0, 5.0, 0.1, 0.01)

# Alpha slider (Elastic Net only)
alpha_val = 0.5
if model_type == "Elastic Net":
    alpha_val = st.sidebar.slider("α (L1/L2 mix)", 0.0, 1.0, 0.5, 0.01)

# Noise injection
noise_val = st.sidebar.slider("Noise (Add to target)", 0.0, 5.0, 0.0, 0.1)

# ------------------------------
# Preprocessing
# ------------------------------
categorical = ["sex", "smoker", "region"]
numerical = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", StandardScaler(), numerical)
])

X = df.drop("charges", axis=1)
y = df["charges"] + np.random.normal(0, noise_val*1000, size=len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Model Selection
# ------------------------------
if model_type == "OLS (Linear Regression)":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=lambda_val)
elif model_type == "Lasso":
    model = Lasso(alpha=lambda_val)
elif model_type == "Elastic Net":
    model = ElasticNet(alpha=lambda_val, l1_ratio=alpha_val)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# ------------------------------
# Metrics
# ------------------------------
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

st.subheader("Model Metrics")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# ------------------------------
# Predicted vs Actual Scatter Plot
# ------------------------------
st.subheader("Predicted vs Actual Scatter Plot")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=y_test, y=pred, mode='markers',
                          marker=dict(color='lightblue', size=8), name='Predictions'))
fig1.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines',
                          line=dict(color='red', dash='dash'), name='Ideal Line'))
fig1.update_layout(xaxis_title="Actual Charges", yaxis_title="Predicted Charges", 
                   plot_bgcolor="#F5F9FF", paper_bgcolor="#F5F9FF")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# Regularization Path (Weights vs λ)
# ------------------------------
st.subheader("Regularization Path (Weights vs λ)")

# Compute weight paths for lambda range
lambda_range = np.linspace(0, 5, 50)
coefs = []

for lam in lambda_range:
    if model_type == "Ridge":
        temp_model = Ridge(alpha=lam)
    elif model_type == "Lasso":
        temp_model = Lasso(alpha=lam)
    elif model_type == "Elastic Net":
        temp_model = ElasticNet(alpha=lam, l1_ratio=alpha_val)
    else:
        temp_model = LinearRegression()
    temp_pipe = Pipeline([("prep", preprocessor), ("model", temp_model)])
    temp_pipe.fit(X_train, y_train)
    coefs.append(temp_pipe.named_steps["model"].coef_)

coefs = np.array(coefs)
feature_names = list(pipe.named_steps["prep"].get_feature_names_out())

fig2 = go.Figure()
for i, name in enumerate(feature_names):
    fig2.add_trace(go.Scatter(x=lambda_range, y=coefs[:, i], mode='lines', name=name))
fig2.update_layout(xaxis_title="λ", yaxis_title="Coefficient Value",
                   plot_bgcolor="#F5F9FF", paper_bgcolor="#F5F9FF")
st.plotly_chart(fig2, use_container_width=True)

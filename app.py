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
from scipy.stats import norm, laplace

# ------------------------------
# Page Config + Theme
# ------------------------------
st.set_page_config(page_title="Interactive Regression Lab", layout="wide", page_icon="📊")

# CSS for peach + brown theme
st.markdown("""
<style>
body {background-color: #FFE5B4;}
.sidebar .sidebar-content {background-color: #FFDAB9;}
h1, h2, h3, h4, h5, h6 {color:#5C4033;}
.stMetric-label, .stMarkdown {color:#5C4033;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Interactive Regression Lab")
st.write("Learn OLS (MLE) and Ridge/Lasso/Elastic Net (MAP) interactively with dynamic visualizations.")

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
                                  ["OLS (MLE)", "Ridge (MAP)", "Lasso (MAP)", "Elastic Net (MAP)"])

# Lambda slider
lambda_val = st.sidebar.slider("λ (Regularization strength)", 0.0, 5.0, 0.1, 0.01)

# Alpha slider (Elastic Net only)
alpha_val = 0.5
if model_type == "Elastic Net (MAP)":
    alpha_val = st.sidebar.slider("α (L1/L2 mix)", 0.0, 1.0, 0.5, 0.01)

# Noise slider
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
if model_type == "OLS (MLE)":
    model = LinearRegression()
elif model_type == "Ridge (MAP)":
    model = Ridge(alpha=lambda_val)
elif model_type == "Lasso (MAP)":
    model = Lasso(alpha=lambda_val)
elif model_type == "Elastic Net (MAP)":
    model = ElasticNet(alpha=lambda_val, l1_ratio=alpha_val)

pipe = Pipeline([("prep", preprocessor), ("model", model)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# ------------------------------
# Metrics
# ------------------------------
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

st.subheader("Model Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MSE", f"{mse:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# ------------------------------
# Dynamic Regression Equation
# ------------------------------
st.subheader("Regression Equation")
feature_names = list(pipe.named_steps["prep"].get_feature_names_out())
weights = pipe.named_steps["model"].coef_
intercept = pipe.named_steps["model"].intercept_

eq_text = f"y = {intercept:.2f}"
for f, w in zip(feature_names, weights):
    eq_text += f" + ({w:.2f} * {f})"
st.code(eq_text)

# ------------------------------
# Predicted vs Actual Scatter
# ------------------------------
st.subheader("Predicted vs Actual Scatter Plot")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=y_test, y=pred, mode='markers',
                          marker=dict(color='brown', size=8), name='Predictions'))
fig1.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines',
                          line=dict(color='darkred', dash='dash'), name='Ideal'))
fig1.update_layout(xaxis_title="Actual Charges", yaxis_title="Predicted Charges",
                   plot_bgcolor="#FFE5B4", paper_bgcolor="#FFE5B4")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# Regularization Path (Weights vs λ)
# ------------------------------
st.subheader("Regularization Path (Weights vs λ)")
lambda_range = np.linspace(0, 5, 50)
coef_paths = []

for lam in lambda_range:
    if model_type == "Ridge (MAP)":
        temp_model = Ridge(alpha=lam)
    elif model_type == "Lasso (MAP)":
        temp_model = Lasso(alpha=lam)
    elif model_type == "Elastic Net (MAP)":
        temp_model = ElasticNet(alpha=lam, l1_ratio=alpha_val)
    else:
        temp_model = LinearRegression()
    temp_pipe = Pipeline([("prep", preprocessor), ("model", temp_model)])
    temp_pipe.fit(X_train, y_train)
    coef_paths.append(temp_pipe.named_steps["model"].coef_)

coef_paths = np.array(coef_paths)
fig2 = go.Figure()
for i, fname in enumerate(feature_names):
    fig2.add_trace(go.Scatter(x=lambda_range, y=coef_paths[:, i], mode='lines', name=fname))
fig2.update_layout(xaxis_title="λ", yaxis_title="Coefficient Value",
                   plot_bgcolor="#FFE5B4", paper_bgcolor="#FFE5B4")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Gaussian Distributions (MLE vs MAP)
# ------------------------------
st.subheader("Gaussian Distributions: Likelihood, Prior, Posterior")
fig3 = go.Figure()
# Likelihood (error distribution)
residuals = y_test - pred
fig3.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Likelihood", marker_color='orange'))
# Prior (for Ridge/Lasso/ElasticNet)
prior_std = np.std(weights) if len(weights)>0 else 1
x_vals = np.linspace(-3*prior_std, 3*prior_std, 100)
if model_type == "Ridge (MAP)" or model_type == "Elastic Net (MAP)":
    prior_vals = norm.pdf(x_vals, loc=0, scale=prior_std)
    fig3.add_trace(go.Scatter(x=x_vals, y=prior_vals*len(weights)*2, mode='lines', name="Prior (Gaussian)", line=dict(color='green')))
elif model_type == "Lasso (MAP)":
    prior_vals = laplace.pdf(x_vals, loc=0, scale=prior_std)
    fig3.add_trace(go.Scatter(x=x_vals, y=prior_vals*len(weights)*2, mode='lines', name="Prior (Laplace)", line=dict(color='green')))
fig3.update_layout(barmode='overlay', xaxis_title='Value', yaxis_title='Frequency', plot_bgcolor="#FFE5B4", paper_bgcolor="#FFE5B4")
fig3.update_traces(opacity=0.7)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Teach-Me Mode
# ------------------------------
st.subheader("📘 Teach-Me Mode")
st.info("""
**What you are seeing:**

- **OLS = MLE** under Gaussian noise assumption.
- **Ridge / Lasso / Elastic Net = MAP** with different priors:
  - Ridge → Gaussian prior → shrinks weights smoothly
  - Lasso → Laplace prior → makes some weights exactly 0 (sparse)
  - Elastic Net → combination of both
- **λ** controls regularization strength.
- **α** controls mix of L1 and L2 in Elastic Net.
- Predicted vs Actual shows accuracy.
- Regularization path shows weight shrinkage.
- Gaussian distributions visualize Likelihood, Prior, and Posterior.
""")

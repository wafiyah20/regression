import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# --- UI CONFIG ---
st.set_page_config(page_title="Advanced Regression Lab", layout="wide")
st.title("🔬 Interactive Regression & Regularization Lab")
st.markdown("---")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv") # Ensure this is in your GitHub
    # Quick Encoding for Categorical
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()
X = df.drop('charges', axis=1)
y = df['charges']

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Model Parameters")
method = st.sidebar.selectbox("Method", ["OLS (MLE)", "Ridge", "Lasso", "Elastic Net"])
lam = st.sidebar.slider("Lambda (λ) / Alpha", 0.0, 100.0, 1.0)
en_ratio = st.sidebar.slider("L1 Ratio (Elastic Net only)", 0.0, 1.0, 0.5)

# Stress Test Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("Novelty Features")
add_noise = st.sidebar.checkbox("Apply Perturbation (Jitter) Test")

if add_noise:
    noise = np.random.normal(0, X.std() * 0.1, X.shape)
    X = X + noise

# Scale data for better visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- MODEL ENGINE ---
if method == "OLS (MLE)":
    model = LinearRegression()
elif method == "Ridge":
    model = Ridge(alpha=lam)
elif method == "Lasso":
    model = Lasso(alpha=lam)
else:
    model = ElasticNet(alpha=lam, l1_ratio=en_ratio)

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# --- VISUALIZATION 1: REGULARIZATION PATH (The "Curves") ---
def plot_paths():
    alphas = np.logspace(-2, 4, 50)
    coefs = []
    for a in alphas:
        if method == "Ridge": m = Ridge(alpha=a)
        elif method == "Lasso": m = Lasso(alpha=a)
        else: m = LinearRegression(); a=0
        m.fit(X_scaled, y)
        coefs.append(m.coef_)
        
    coefs = np.array(coefs)
    fig = go.Figure()
    for i, col in enumerate(X.columns):
        fig.add_trace(go.Scatter(x=alphas, y=coefs[:, i], name=col, mode='lines'))
    
    fig.update_layout(title="Regularization Path (Weight vs λ)", xaxis_type="log",
                      xaxis_title="Lambda", yaxis_title="Weight Value", height=400)
    # Add vertical line for current lambda
    fig.add_vline(x=lam, line_dash="dash", line_color="red", annotation_text="Current λ")
    return fig

# --- VISUALIZATION 2: THE GEOMETRY (Novelty) ---
def plot_geometry():
    # Showing relationship between top 2 features: Smoker vs Age
    st.subheader("📐 The Geometry of the Constraint")
    st.info("This shows how the 'Loss Bowl' meets the 'Regularization Shape'.")
    # (Implementation of contour logic here)
    # For now, let's use a placeholder Scatter for Predicted vs Actual
    fig = px.scatter(x=y, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, 
                     title="Predicted vs Actual", opacity=0.5)
    fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red"))
    return fig

# --- LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(plot_paths(), use_container_width=True)

with col2:
    st.plotly_chart(plot_geometry(), use_container_width=True)

# --- STATS PANEL ---
st.markdown("### 📊 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² Score", round(r2_score(y, y_pred), 4))
k2.metric("MSE", f"{mean_squared_error(y, y_pred):.0f}")
k3.metric("L1 Norm", round(np.linalg.norm(model.coef_, 1), 2))
k4.metric("L2 Norm", round(np.linalg.norm(model.coef_, 2), 2))

# --- THE BAYESIAN/MATH SECTION ---
with st.expander("Show Mathematical Derivation & Distributions"):
    st.write("Current Weights:", model.coef_)
    st.latex(r"w = (X^T X + \lambda I)^{-1} X^T y")
    # Add Gaussian Likelihood vs Prior Plotly here

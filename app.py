import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- STYLING & THEME ---
st.set_page_config(page_title="Regression Lab Pro", layout="wide")

# Custom CSS for Light Theme & Smooth UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE ENGINE ---
@st.cache_data
def get_data():
    # Use your insurance.csv path
    df = pd.read_csv("insurance.csv")
    for col in ['sex', 'smoker', 'region']:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

df = get_data()
target = 'charges'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Scale features (Critical for Regularization visualization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SIDEBAR: INTERACTIVE CONTROLS ---
st.sidebar.title("🛠️ Model Studio")
method = st.sidebar.radio("Select Strategy", ["OLS (MLE)", "Ridge (L2)", "Lasso (L1)", "Elastic Net"])
lam = st.sidebar.slider("Regularization Strength (λ)", 0.0, 20.0, 1.0, 0.1)

# NOVELTY: The Jitter Test
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Tests")
jitter = st.sidebar.checkbox("🧬 Enable Stability Test (Jitter)")
if jitter:
    X_scaled = X_scaled + np.random.normal(0, 0.05, X_scaled.shape)

# --- MODEL SELECTION LOGIC ---
if method == "OLS (MLE)":
    model = LinearRegression()
elif method == "Ridge (L2)":
    model = Ridge(alpha=lam)
elif method == "Lasso (L1)":
    model = Lasso(alpha=lam)
else:
    l1_rat = st.sidebar.slider("L1 Ratio (α)", 0.0, 1.0, 0.5)
    model = ElasticNet(alpha=lam, l1_ratio=l1_rat)

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
weights = model.coef_

# --- VISUALIZATION 1: INTERACTIVE WEIGHT PATH ---
st.subheader("📈 The Weight Shrinkage Journey")
st.caption("Hover over the curves to see which features 'survive' regularization.")

# Pre-calculate path for the graph background
alphas = np.linspace(0.01, 20, 50)
path_data = []
for a in alphas:
    if "Ridge" in method: m = Ridge(alpha=a)
    elif "Lasso" in method: m = Lasso(alpha=a)
    else: m = LinearRegression()
    m.fit(X_scaled, y)
    path_data.append(m.coef_)

path_df = np.array(path_data)
fig_path = go.Figure()

for i, col in enumerate(features):
    fig_path.add_trace(go.Scatter(x=alphas, y=path_df[:, i], name=col,
                                 line=dict(width=2), hovertemplate=f"<b>{col}</b><br>λ: %{{x}}<br>Weight: %{{y:.2f}}"))

# Vertical indicator for current Lambda
fig_path.add_vline(x=lam, line_dash="dash", line_color="#ff4b4b", annotation_text="Active λ")
fig_path.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_path, use_container_width=True)

# --- VISUALIZATION 2: PREDICTED VS ACTUAL (WITH RESIDUALS) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Prediction Accuracy")
    fig_res = px.scatter(x=y, y=y_pred, color=np.abs(y-y_pred),
                         color_continuous_scale='Viridis', labels={'x': 'Actual Charges', 'y': 'Predicted'},
                         opacity=0.6)
    fig_res.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dot"))
    fig_res.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig_res, use_container_width=True)

with col2:
    st.subheader("💎 Weight Sparsity")
    # Horizontal bar chart of weights
    weight_df = pd.DataFrame({'Feature': features, 'Weight': weights}).sort_values('Weight')
    fig_weights = px.bar(weight_df, x='Weight', y='Feature', orientation='h',
                         color='Weight', color_continuous_scale='RdBu_r')
    fig_weights.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_weights, use_container_width=True)

# --- NOVELTY: THE GEOMETRY OF CONSTRAINTS (Visual Explanation) ---

with st.expander("🔮 Advanced Theory: Why is Lasso sparse?"):
    st.write("In Lasso (L1), the constraint is a **Diamond**. The Loss Contours are more likely to hit the 'corners' of the diamond, which are exactly on the axes (where one weight becomes zero). In Ridge (L2), the constraint is a **Circle**, which doesn't have corners, so weights stay small but rarely hit zero.")

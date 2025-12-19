import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Regression", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🎮 Model Controls")
method = st.sidebar.selectbox("Regression Strategy", 
                            ["MLE (OLS)", "MAP (Ridge)", "MAP (Lasso)", "Elastic Net"])

# Regularization Strength Slider (10^x for smooth responsiveness)
exp = st.sidebar.slider("Regularization Strength (λ)", -2.0, 4.0, 0.0, 0.1)
lam = 10**exp

l1_ratio = 0.5
if method == "Elastic Net":
    l1_ratio = st.sidebar.slider("Mixing Ratio (L1 vs L2)", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
jitter_active = st.sidebar.checkbox(" Enable Jitter (Stability Test)")

# --- 4. MODEL ENGINE ---
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
X = df[features].values
y = df['charges'].values

if jitter_active:
    X = X + np.random.normal(0, X.std(axis=0) * 0.1, X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if method == "MLE (OLS)":
    model = LinearRegression()
    lam_eff = 0
elif method == "MAP (Ridge)":
    model = Ridge(alpha=lam)
    lam_eff = lam
elif method == "MAP (Lasso)":
    model = Lasso(alpha=lam)
    lam_eff = lam
else:
    model = ElasticNet(alpha=lam, l1_ratio=l1_ratio)
    lam_eff = lam

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# --- 5. TABS ---
st.title("Regression Analysis")
st.markdown(f"**Current Approach:** {method} | **Effective λ:** {lam_eff:.4f}")
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "🌌 3D Optimization Surface", "📋 Data Statistics"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Weight Shrinkage Path")
        alphas_path = np.logspace(-2, 4, 100)
        path_coefs = []
        for a in alphas_path:
            if "Ridge" in method: m = Ridge(alpha=a)
            elif "Lasso" in method: m = Lasso(alpha=a)
            elif "Elastic" in method: m = ElasticNet(alpha=a, l1_ratio=l1_ratio)
            else: m = LinearRegression(); a=0
            m.fit(X_scaled, y)
            path_coefs.append(m.coef_)
        
        fig_path = go.Figure()
        path_coefs = np.array(path_coefs)
        for i, name in enumerate(features):
            fig_path.add_trace(go.Scatter(x=alphas_path, y=path_coefs[:, i], name=name))
        fig_path.add_vline(x=lam_eff, line_dash="dash", line_color="red")
        fig_path.update_layout(xaxis_type="log", template="plotly_white", height=350, 
                              xaxis_title="Lambda", yaxis_title="Weight")
        st.plotly_chart(fig_path, use_container_width=True)

    with col2:
        st.subheader("Feature Weights")
        w_df = pd.DataFrame({'Feature': features, 'Value': model.coef_}).sort_values('Value')
        fig_w = px.bar(w_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='RdBu_r')
        fig_w.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
    m2.metric("MSE (Error)", f"{mean_squared_error(y, y_pred):.0e}")
    m3.metric("L1 (Sparsity)", f"{np.linalg.norm(model.coef_, 1):.1f}")
    m4.metric("L2 (Smoothness)", f"{np.linalg.norm(model.coef_, 2):.1f}")

    # THE REQUESTED SCATTER PLOT
    st.subheader("Predicted vs Actual (Model Fit)")
    fig_pa = px.scatter(x=y, y=y_pred, color=np.abs(y-y_pred), opacity=0.4, color_continuous_scale='Viridis',
                         labels={'x': 'Actual Charges', 'y': 'Predicted Charges'})
    # Perfect Prediction Line
    fig_pa.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dot"))
    fig_pa.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_pa, use_container_width=True)

with tab2:
    st.header("The 3D Optimization Landscape")
    
    st.markdown("""
    **What are you looking at?**
    * Imagine this is a landscape. The model is a ball trying to roll to the lowest point (the blue/purple area).
    * **λ (Lambda)** acts like a force that changes the shape of the land. 
    * **In Lasso (L1)**, notice how the bottom becomes "pointy" or diamond-like. This "pointiness" is what traps weights at zero.
    * **In Ridge (L2)**, the shape stays a smooth, circular bowl, just steeper.
    """)

    # 3D Surface Logic
    w_grid = np.linspace(-2, 2, 50)
    W1, W2 = np.meshgrid(w_grid, w_grid)
    RSS = (W1 - 1)**2 + (W2 - 1)**2 
    
    if "Ridge" in method:
        penalty = 0.5 * lam * (W1**2 + W2**2)
        c_scale = 'Viridis'
    elif "Lasso" in method:
        penalty = 2 * lam * (np.abs(W1) + np.abs(W2))
        c_scale = 'Hot'
    elif "Elastic" in method:
        penalty = lam * (l1_ratio * (np.abs(W1) + np.abs(W2)) + (1-l1_ratio) * (W1**2 + W2**2))
        c_scale = 'Electric'
    else:
        penalty = 0
        c_scale = 'Blues'
        
    Z = RSS + penalty
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=w_grid, y=w_grid, colorscale=c_scale)])
    fig_3d.update_layout(scene=dict(xaxis_title='Weight A', yaxis_title='Weight B', zaxis_title='Total Loss'),
                         height=700, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("Insurance Data Profile")
    st.dataframe(df.describe().T, use_container_width=True)
    st.subheader("Correlation Heatmap")
    st.plotly_chart(px.imshow(df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

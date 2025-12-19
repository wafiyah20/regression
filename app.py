import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Regression Lab Pro", layout="wide")

# Custom CSS for a clean, light look
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
method = st.sidebar.selectbox("Regression Method", ["OLS (None)", "Ridge (L2)", "Lasso (L1)", "Elastic Net"])
exp = st.sidebar.slider("Regularization Strength (10^x)", -2.0, 4.0, 0.0, 0.1)
lam = 10**exp

st.sidebar.markdown("---")
jitter_active = st.sidebar.checkbox("🧬 Enable Jitter (Stability Test)")

# --- 4. MODEL ENGINE ---
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
X = df[features].values
y = df['charges'].values

if jitter_active:
    X = X + np.random.normal(0, X.std(axis=0) * 0.08, X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if method == "OLS (None)":
    model = LinearRegression()
elif method == "Ridge (L2)":
    model = Ridge(alpha=lam)
elif method == "Lasso (L1)":
    model = Lasso(alpha=lam)
else:
    model = ElasticNet(alpha=lam, l1_ratio=0.5)

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# --- 5. TABS ---
st.title("🔬 Advanced Regression Analysis Lab")
tab1, tab2, tab3 = st.tabs(["📊 Performance", "🌌 3D Landscape Explained", "📋 Data Stats"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Weight Shrinkage Path")
        alphas_path = np.logspace(-2, 4, 100)
        path_coefs = []
        for a in alphas_path:
            if "Ridge" in method: m = Ridge(alpha=a)
            elif "Lasso" in method: m = Lasso(alpha=a)
            elif "Elastic" in method: m = ElasticNet(alpha=a, l1_ratio=0.5)
            else: m = LinearRegression(); a=0
            m.fit(X_scaled, y)
            path_coefs.append(m.coef_)
        
        fig_path = go.Figure()
        path_coefs = np.array(path_coefs)
        for i, name in enumerate(features):
            fig_path.add_trace(go.Scatter(x=alphas_path, y=path_coefs[:, i], name=name))
        fig_path.add_vline(x=lam, line_dash="dash", line_color="red")
        fig_path.update_layout(xaxis_type="log", template="plotly_white", height=400)
        st.plotly_chart(fig_path, use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        w_df = pd.DataFrame({'Feature': features, 'Weight': model.coef_}).sort_values('Weight')
        fig_w = px.bar(w_df, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='RdBu_r')
        fig_w.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
    m2.metric("MSE (Error)", f"{mean_squared_error(y, y_pred):.1e}")
    m3.metric("Complexity (L1)", f"{np.linalg.norm(model.coef_, 1):.1f}")
    m4.metric("Complexity (L2)", f"{np.linalg.norm(model.coef_, 2):.1f}")

with tab2:
    st.header("The 3D Loss Landscape")
    
    # Simple bullet points for others to understand
    st.markdown("""
    **How to read this graph:**
    * **The Valley:** The 'Error' the model makes. Lower is better.
    * **The Surface:** Shows how the model searches for the best 'Weights' (W1 and W2).
    * **The Effect of λ:** As you increase the slider, you are adding a 'Penalty' that warps this shape.
    * **Why Lasso is Special:** Notice the 'sharp' edges in Lasso. This forces the model to choose exactly zero for some weights!
    """)

    # 3D Math Logic
    w1_range = np.linspace(-2, 2, 50)
    w2_range = np.linspace(-2, 2, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    RSS = (W1 - 1)**2 + (W2 - 1)**2 # The 'Goal' Bowl
    
    if "Ridge" in method:
        penalty = 0.5 * lam * (W1**2 + W2**2)
        color_scale = 'Viridis'
    elif "Lasso" in method:
        penalty = 2 * lam * (np.abs(W1) + np.abs(W2))
        color_scale = 'Hot'
    else:
        penalty = 0
        color_scale = 'Blues'
        
    Z = RSS + penalty
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=w1_range, y=w2_range, colorscale=color_scale)])
    fig_3d.update_layout(scene=dict(xaxis_title='Weight A', yaxis_title='Weight B', zaxis_title='Total Loss (Error + Penalty)'), 
                         height=700, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("Data Overview")
    st.write(df.describe())
    st.subheader("Correlations")
    st.plotly_chart(px.imshow(df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

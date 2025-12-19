import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. PAGE CONFIG & CUSTOM STYLES ---
st.set_page_config(page_title="Regression Lab Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #f1f5f9; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PRE-PROCESSING ---
@st.cache_data
def load_data():
    # Make sure insurance.csv is in your root directory
    df = pd.read_csv("insurance.csv")
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Model Configuration")
method = st.sidebar.selectbox("Regression Method", 
                            ["OLS (MLE)", "Ridge (L2)", "Lasso (L1)", "Elastic Net"])

# We use an exponent slider to make Lambda highly responsive
exp = st.sidebar.slider("Regularization Strength (10^x)", -2.0, 4.0, 0.0, 0.1)
lam = 10**exp

en_ratio = 0.5
if method == "Elastic Net":
    en_ratio = st.sidebar.slider("L1/L2 Mix (Alpha)", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Novelty: Stability Test")
jitter_active = st.sidebar.checkbox("Apply Feature Jitter")

# --- 4. MODEL ENGINE ---
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
X = df[features].values
y = df['charges'].values

if jitter_active:
    X = X + np.random.normal(0, X.std(axis=0) * 0.05, X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Selection
if method == "OLS (MLE)":
    model = LinearRegression()
    lam_display = 0
elif method == "Ridge (L2)":
    model = Ridge(alpha=lam)
    lam_display = lam
elif method == "Lasso (L1)":
    model = Lasso(alpha=lam)
    lam_display = lam
else:
    model = ElasticNet(alpha=lam, l1_ratio=en_ratio)
    lam_display = lam

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# --- 5. TABS INTERFACE ---
st.title("🔬 Advanced Regression Analysis Lab")
st.markdown(f"**Strategy:** {method} | **λ Value:** {lam_display:.4f}")

tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "📐 Geometry & Probabilistic Math", "📋 Data Insights"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Weight Shrinkage Path")
        alphas_path = np.logspace(-2, 4, 100)
        path_coefs = []
        for a in alphas_path:
            if "Ridge" in method: m = Ridge(alpha=a)
            elif "Lasso" in method: m = Lasso(alpha=a)
            elif "Elastic" in method: m = ElasticNet(alpha=a, l1_ratio=en_ratio)
            else: m = LinearRegression(); a=0
            m.fit(X_scaled, y)
            path_coefs.append(m.coef_)
        
        path_coefs = np.array(path_coefs)
        fig_path = go.Figure()
        for i, name in enumerate(features):
            fig_path.add_trace(go.Scatter(x=alphas_path, y=path_coefs[:, i], name=name))
        
        fig_path.add_vline(x=lam, line_dash="dash", line_color="red")
        fig_path.update_layout(xaxis_type="log", template="plotly_white", height=400,
                              xaxis_title="Lambda (Log Scale)", yaxis_title="Weight Magnitude")
        st.plotly_chart(fig_path, use_container_width=True)

    with col2:
        st.subheader("Current Weights")
        w_df = pd.DataFrame({'Feature': features, 'Value': model.coef_}).sort_values('Value')
        fig_w = px.bar(w_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='RdBu_r')
        fig_w.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2_score(y, y_pred):.4f}")
    m2.metric("MSE", f"{mean_squared_error(y, y_pred):.0e}")
    m3.metric("L1 Norm (Sparsity)", f"{np.linalg.norm(model.coef_, 1):.2f}")
    m4.metric("L2 Norm (Smoothness)", f"{np.linalg.norm(model.coef_, 2):.2f}")

    # Pred vs Actual
    st.subheader("Residual Analysis (Predicted vs Actual)")
    fig_pa = px.scatter(x=y, y=y_pred, color=np.abs(y-y_pred), opacity=0.5, color_continuous_scale='Viridis')
    fig_pa.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dot"))
    fig_pa.update_layout(template="plotly_white", xaxis_title="Actual Charges", yaxis_title="Predicted Charges")
    st.plotly_chart(fig_pa, use_container_width=True)

with tab2:
    with tab2:
    st.markdown("---")
    st.header("🌌 3D Loss Landscape")
    st.write("This 3D surface shows the Error (MSE) + Penalty. The model's goal is to find the lowest point.")
    
    # Create grid for 3D
    w1_3d = np.linspace(-2, 2, 50)
    w2_3d = np.linspace(-2, 2, 50)
    W1, W2 = np.meshgrid(w1_3d, w2_3d)
    
    # Base Loss (The Bowl)
    RSS = (W1 - 1)**2 + (W2 - 1)**2
    
    # Add Penalty
    if "Ridge" in method:
        penalty = lam * (W1**2 + W2**2)
        title_3d = "Ridge: Smooth Paraboloid"
    elif "Lasso" in method:
        penalty = lam * (np.abs(W1) + np.abs(W2))
        title_3d = "Lasso: Sharp 'V' Shape at Axes"
    else:
        penalty = 0
        title_3d = "OLS: Simple Quadratic Bowl"
        
    Z = RSS + penalty

    fig_3d = go.Figure(data=[go.Surface(z=Z, x=w1_3d, y=w2_3d, colorscale='Viridis')])
    
    fig_3d.update_layout(
        title=title_3d,
        scene=dict(
            xaxis_title='Weight 1',
            yaxis_title='Weight 2',
            zaxis_title='Total Loss',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.header("The Geometry of Optimization")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### Weight Space Constraint")
        # Visualizing the L1 vs L2 "Fence"
        w_grid = np.linspace(-2, 2, 100)
        w1, w2 = np.meshgrid(w_grid, w_grid)
        loss_bowl = (w1 - 1)**2 + (w2 - 1)**2 # Theoretical loss
        
        fig_geo = go.Figure()
        fig_geo.add_trace(go.Contour(z=loss_bowl, x=w_grid, y=w_grid, colorscale='Blues', opacity=0.3, showscale=False))
        
        # Scaling the 'fence' size based on lambda
        t = 1.5 / (1 + 0.5*np.log10(lam + 1)) 
        
        if "Lasso" in method:
            fig_geo.add_shape(type="path", path=f"M 0,{t} L {t},0 L 0,{-t} L {-t},0 Z", line_color="Red", fillcolor="rgba(255,0,0,0.2)")
            st.info("Lasso (L1) creates a diamond. Notice how the 'corners' on the axes encourage weights to hit zero.")
        else:
            fig_geo.add_shape(type="circle", x0=-t, y0=-t, x1=t, y1=t, line_color="Green", fillcolor="rgba(0,255,0,0.2)")
            st.info("Ridge (L2) creates a circle. This keeps weights small but rarely hits exactly zero.")
        
        fig_geo.update_layout(width=500, height=500, template="plotly_white", xaxis_title="Weight 1", yaxis_title="Weight 2")
        st.plotly_chart(fig_geo)

    with g_col2:
        st.markdown("#### Probabilistic View (MAP Estimation)")
        x_dist = np.linspace(-4, 4, 100)
        likelihood = np.exp(-0.5 * (x_dist - 1.5)**2)
        # Prior narrows as lambda increases
        prior_width = 1.0 / (0.5 + 0.2*np.log10(lam + 1))
        prior = np.exp(-0.5 * (x_dist / prior_width)**2)
        posterior = likelihood * prior
        
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=x_dist, y=likelihood, name="Likelihood (MLE)", fill='tozeroy'))
        fig_prob.add_trace(go.Scatter(x=x_dist, y=prior, name="Prior (Regularizer)", fill='tozeroy'))
        fig_prob.add_trace(go.Scatter(x=x_dist, y=posterior, name="Posterior (MAP)", line=dict(color='black', width=3)))
        fig_prob.update_layout(template="plotly_white", title="Weights as Distributions")
        st.plotly_chart(fig_prob)

    st.markdown("---")
    st.subheader("Mathematical Derivation")
    st.latex(r"\hat{w} = \text{arg min}_w \|y - Xw\|^2_2 + \lambda \text{Penalty}(w)")
    st.latex(r"\text{Ridge Penalty: } \|w\|^2_2 \quad \text{Lasso Penalty: } \|w\|_1")

with tab3:
    st.subheader("Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

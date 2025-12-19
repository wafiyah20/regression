import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. PAGE CONFIG & STYLE ---
st.set_page_config(page_title="Advanced Regression Lab", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    h1, h2, h3 { color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_clean_data():
    # Attempt to load insurance dataset
    try:
        df = pd.read_csv("insurance.csv")
    except:
        st.error("Please ensure 'insurance.csv' is in your GitHub folder.")
        return None
    
    # Feature Engineering for visualization
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_and_clean_data()

if df is not None:
    # --- 3. SIDEBAR CONTROLS ---
    st.sidebar.title("🎮 Control Panel")
    
    method = st.sidebar.selectbox("Regression Method", 
                                ["OLS (MLE)", "Ridge (MAP - L2)", "Lasso (MAP - L1)", "Elastic Net"])
    
    # Logarithmic Slider for λ (The Fix for 'No Movement')
    # This allows λ to range from 0.01 to 1000 effectively
    exp = st.sidebar.slider("Regularization Strength (10^x)", -2.0, 4.0, 0.0, 0.1)
    lam = 10**exp
    
    alpha_en = 0.5
    if method == "Elastic Net":
        alpha_en = st.sidebar.slider("Elastic Net α (L1 vs L2 ratio)", 0.0, 1.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Creativity: Stress Tests")
    jitter_mode = st.sidebar.checkbox("🧬 Apply Jitter (Stability Test)")
    
    # --- 4. PRE-PROCESSING ---
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    target = 'charges'
    
    X = df[features].values
    y = df[target].values
    
    # Novelty: The Jitter Test
    if jitter_mode:
        noise = np.random.normal(0, X.std(axis=0) * 0.05, X.shape)
        X = X + noise

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 5. MODEL ENGINE ---
    if method == "OLS (MLE)":
        model = LinearRegression()
    elif method == "Ridge (MAP - L2)":
        model = Ridge(alpha=lam)
    elif method == "Lasso (MAP - L1)":
        model = Lasso(alpha=lam)
    else:
        model = ElasticNet(alpha=lam, l1_ratio=alpha_en)

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    # --- 6. LAYOUT: VISUALIZATIONS ---
    st.title("🔬 Advanced Regression Lab")
    st.markdown(f"Currently viewing: **{method}** | λ = **{lam:.4f}**")

    tab1, tab2, tab3 = st.tabs(["📈 Main Dashboard", "📐 Geometry & Math", "📋 Data Statistics"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # INTERACTIVE REGULARIZATION PATH
            st.subheader("Weight Shrinkage Path")
            alphas_path = np.logspace(-2, 4, 100)
            coefs = []
            for a in alphas_path:
                if "Ridge" in method: m = Ridge(alpha=a)
                elif "Lasso" in method: m = Lasso(alpha=a)
                elif "Elastic" in method: m = ElasticNet(alpha=a, l1_ratio=alpha_en)
                else: m = LinearRegression(); a=0
                m.fit(X_scaled, y)
                coefs.append(m.coef_)
            
            coefs = np.array(coefs)
            fig_path = go.Figure()
            for i, name in enumerate(features):
                fig_path.add_trace(go.Scatter(x=alphas_path, y=coefs[:, i], name=name,
                                             mode='lines', line=dict(width=2)))
            
            fig_path.add_vline(x=lam, line_dash="dash", line_color="red", annotation_text="Active λ")
            fig_path.update_layout(xaxis_type="log", template="plotly_white", 
                                  xaxis_title="Lambda (Log Scale)", yaxis_title="Coefficient Value")
            st.plotly_chart(fig_path, use_container_width=True)

        with col2:
            # WEIGHT IMPORTANCE BAR
            st.subheader("Feature Weights")
            weight_df = pd.DataFrame({'Feature': features, 'Weight': model.coef_})
            fig_bar = px.bar(weight_df, x='Weight', y='Feature', orientation='h',
                             color='Weight', color_continuous_scale='RdBu_r')
            fig_bar.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # METRICS ROW
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
        m2.metric("MAE", f"{mean_absolute_error(y, y_pred):.0f}")
        m3.metric("MSE", f"{mean_squared_error(y, y_pred):.0e}")
        m4.metric("L1 Norm", f"{np.linalg.norm(model.coef_, 1):.2f}")
        m5.metric("L2 Norm", f"{np.linalg.norm(model.coef_, 2):.2f}")

        # PREDICTED VS ACTUAL
        st.subheader("Predicted vs Actual Analysis")
        fig_pred = px.scatter(x=y, y=y_pred, opacity=0.4, color=np.abs(y-y_pred),
                             labels={'x': 'Actual Charges', 'y': 'Predicted Charges'},
                             color_continuous_scale='ice')
        fig_pred.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dash"))
        fig_pred.update_layout(template="plotly_white")
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        # GEOMETRY OF REGULARIZATION
        st.subheader("The Probabilistic View (MLE vs MAP)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Gaussian Likelihood & Prior")
            # Visualization of Likelihood (MLE) vs Posterior (MAP)
            x_range = np.linspace(-5, 5, 100)
            prior = np.exp(-0.5 * x_range**2) # Simplified Gaussian Prior
            likelihood = np.exp(-0.5 * (x_range - 2)**2) # Centered at a weight of 2
            posterior = prior * likelihood
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=x_range, y=likelihood, name="Likelihood (MLE)", fill='tozeroy'))
            fig_dist.add_trace(go.Scatter(x=x_range, y=prior, name="Prior (Regularizer)", fill='tozeroy'))
            fig_dist.add_trace(go.Scatter(x=x_range, y=posterior, name="Posterior (MAP)", line=dict(width=4, color='black')))
            fig_dist.update_layout(template="plotly_white", title="Probabilistic Weight Estimation")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with c2:
            st.markdown("#### Step-by-Step Matrix Math")
            st.latex(r"W_{OLS} = (X^T X)^{-1} X^T Y")
            st.latex(r"W_{Ridge} = (X^T X + \lambda I)^{-1} X^T Y")
            st.info("Notice how λ I is added to the diagonal, making the matrix inversion stable.")

    with tab3:
        st.subheader("Dataset Statistics & Correlations")
        st.write(df.describe())
        
        # Correlation Matrix
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation (Rxx & Rxy)")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- 7. FOOTER ---
st.markdown("---")
st.caption("Developed for Advanced Regression Learning | Novel Features: Log-Lambda Scaling & Jitter Stability Analysis")

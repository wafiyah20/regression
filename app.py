import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------------------------------------
# STREAMLIT CONFIG + THEME
# ---------------------------------------------
st.set_page_config(
    page_title="Interactive Regression Lab",
    layout="wide",
    page_icon="📘"
)

# Custom Baby-Blue Theme CSS
st.markdown("""
<style>
body {
    background-color: #E8F3FF;
}
.sidebar .sidebar-content {
    background-color: #D9EBFF !important;
}
.big-title {
    font-size:36px;
    color:#004D99;
    font-weight:700;
    text-align:center;
}
.card {
    background:#FFFFFF;
    padding:20px;
    border-radius:18px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.07);
}
.section-title {
    color:#0066CC;
    font-size:22px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# TITLE
# ---------------------------------------------
st.markdown('<p class="big-title">📘 Interactive Regression Learning Lab</p>', unsafe_allow_html=True)
st.write("A beautiful baby-blue regression simulator that helps beginners learn models visually. 💙")

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()

st.markdown("### 📂 Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------
# PREPROCESSING
# ---------------------------------------------
categorical = ["sex", "smoker", "region"]
numerical = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", StandardScaler(), numerical)
])

# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------
st.sidebar.title("⚙️ Controls")

model_type = st.sidebar.selectbox(
    "Choose Regression Type",
    ["OLS (Linear Regression)", "Ridge", "Lasso", "Elastic Net"]
)

lambda_val = st.sidebar.slider("λ (Regularization Strength)", 0.0, 2.0, 0.1, 0.01)

alpha_val = 0.5
if model_type == "Elastic Net":
    alpha_val = st.sidebar.slider("α (Mixing: L1 vs L2)", 0.0, 1.0, 0.5, 0.01)

noise_amount = st.sidebar.slider("Add Noise for Experiment", 0.0, 5.0, 0.0, 0.1)

# ---------------------------------------------
# TRAIN/TEST SPLIT + MODEL SELECTION
# ---------------------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

# Noise injection
y_noisy = y + np.random.normal(0, noise_amount * 1000, size=len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Model choice
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

# ---------------------------------------------
# METRICS
# ---------------------------------------------
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# ------------------- METRICS CARD --------------------
st.markdown("### 📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("MAE", f"{mae:,.2f}")
col4.metric("R² Score", f"{r2:.4f}")

# ---------------------------------------------
# FEATURE IMPORTANCE HEATMAP (Coefficients)
# ---------------------------------------------
st.markdown("<div class='section-title'>🔥 Feature Importance (Heatmap)</div>", unsafe_allow_html=True)

# Get final feature names
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(categorical)
all_features = list(cat_features) + numerical

coefs = pipe.named_steps["model"].coef_
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap([coefs], annot=True, cmap="Blues", xticklabels=all_features)
plt.yticks([])
st.pyplot(fig)

# ---------------------------------------------
# PREDICTED VS ACTUAL
# ---------------------------------------------
st.markdown("<div class='section-title'>📈 Predicted vs Actual Charges</div>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(y_test, pred, alpha=0.6)
ax2.set_xlabel("Actual Charges")
ax2.set_ylabel("Predicted Charges")
ax2.set_title("Predicted vs Actual")
st.pyplot(fig2)

# ---------------------------------------------
# BIAS–VARIANCE (Changing λ)
# ---------------------------------------------
st.markdown("<div class='section-title'>🔄 Bias–Variance Curve</div>", unsafe_allow_html=True)

lambdas = np.linspace(0, 2, 30)
errors = []

for lam in lambdas:
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
    temp_pred = temp_pipe.predict(X_test)
    errors.append(mean_squared_error(y_test, temp_pred))

fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(lambdas, errors)
ax3.set_xlabel("λ")
ax3.set_ylabel("MSE")
ax3.set_title("Bias–Variance Tradeoff")
st.pyplot(fig3)

# ---------------------------------------------
# TEACH-ME MODE
# ---------------------------------------------
st.markdown("<div class='section-title'>📘 Teach-Me Mode</div>", unsafe_allow_html=True)

st.info("""
### What does λ (lambda) do?
- Higher λ → more shrinkage → simpler model  
- Low λ → flexible model → risk of overfitting  

### What does α (alpha) in Elastic Net do?
- α = 0 → becomes pure Ridge  
- α = 1 → becomes pure Lasso  
- 0 < α < 1 → mix of both  

### Why does Lasso make coefficients 0?
Because L1 regularization creates a sharp corner at zero → it pushes small weights to exactly zero.
""")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e5e7eb;
}
.metric-title {
    color: gray;
    font-size: 14px;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #0f172a;
}
.predict-btn {
    background-color: #2563eb;
    color: white;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("best_model.pkl")

FEATURE_COLUMNS = [
    'BHK', 'Size', 'Bathroom', 'Total_Floors',
    'Area Type_Carpet Area', 'Area Type_Super Area',
    'City_Chennai', 'City_Delhi', 'City_Hyderabad',
    'City_Kolkata', 'City_Mumbai',
    'Furnishing Status_Semi-Furnished',
    'Furnishing Status_Unfurnished',
    'Tenant Preferred_Bachelors/Family',
    'Tenant Preferred_Family'
]

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>🏠 House Rent Price Prediction</h1>
    <p>Advanced Machine Learning Application with Real Estate Data</p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🏡 Property Configuration")

BHK = st.sidebar.slider("BHK", 1, 6, 2)
Size = st.sidebar.slider("Size (sqft)", 200, 5000, 800)
Bathroom = st.sidebar.slider("Bathrooms", 1, 5, 2)
Total_Floors = st.sidebar.slider("Total Floors", 1, 50, 10)

AreaType = st.sidebar.selectbox("Area Type", ["Built Area", "Carpet Area", "Super Area"])
City = st.sidebar.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
Furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
Tenant = st.sidebar.selectbox("Tenant Preferred", ["Bachelors/Family", "Family"])

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Insights", "📁 Dataset"])

# ================= TAB 1: PREDICTION =================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Property Summary")
        st.write(f"""
        **BHK:** {BHK}  
        **Size:** {Size} sqft  
        **Bathrooms:** {Bathroom}  
        **Total Floors:** {Total_Floors}  
        **Area Type:** {AreaType}  
        **City:** {City}  
        **Furnishing:** {Furnishing}  
        **Tenant Preferred:** {Tenant}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 Rent Prediction")

        if st.button("🔮 Predict Rent", use_container_width=True):
            input_df = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)

            input_df['BHK'] = BHK
            input_df['Size'] = Size
            input_df['Bathroom'] = Bathroom
            input_df['Total_Floors'] = Total_Floors

            if AreaType != "Built Area":
                input_df[f'Area Type_{AreaType}'] = 1

            if City != "Bangalore":
                input_df[f'City_{City}'] = 1

            if Furnishing != "Furnished":
                input_df[f'Furnishing Status_{Furnishing}'] = 1

            input_df[f'Tenant Preferred_{Tenant}'] = 1

            log_rent = model.predict(input_df)[0]
            rent = np.exp(log_rent)

            st.success(f"💰 ₹ {rent:,.0f} per month")

        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: INSIGHTS =================
with tab2:
    st.subheader("📊 Quick Insights")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Selected City</div>
        <div class="metric-value">{City}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Property Size</div>
        <div class="metric-value">{Size} sqft</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BHK</div>
        <div class="metric-value">{BHK}</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 Tip: Larger size and metro cities usually result in higher rent.")

# ================= TAB 3: DATASET =================
with tab3:
    st.subheader("📁 Dataset Preview")
    df = pd.read_csv("data/House_Rent_Dataset.csv")
    st.dataframe(df.head(15))

# ================= FOOTER =================
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Built with ❤️ | Streamlit • XGBoost • Machine Learning
</p>
""", unsafe_allow_html=True)

import streamlit as st
from PIL import Image

# ------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ------------------------------------------
st.set_page_config(
    page_title="Brazil's Top Exports | Control Tower",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enforcing "Scrapbook/Natural" Theme with Brazilian Accents
st.markdown("""
<style>
    /* Main Background - Light Paper Texture */
    .stApp {
        background-color: #f4f1ea; 
        color: #2c3e50; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Hero Title Styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1b5e20; /* Dark Green */
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        font-size: 1.5rem;
        color: #5d4037; /* Earthy Brown */
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }

    /* Product Cards */
    .product-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        border-bottom: 5px solid #ddd;
        height: 250px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px; /* Spacing between card and button */
    }
    .product-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* Card Icons */
    .card-icon {
        font-size: 4rem;
        margin-bottom: 10px;
    }

    /* Card Titles */
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Description Text */
    .card-text {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.4;
    }

    /* Custom Border Colors for Cards */
    .btn-soy { border-bottom: 5px solid #2e7d32; }
    .btn-oil { border-bottom: 5px solid #2c3e50; }
    .btn-iron { border-bottom: 5px solid #c62828; }
    .btn-coffee { border-bottom: 5px solid #5d4037; }
    .btn-meat { border-bottom: 5px solid #d32f2f; }

    /* Streamlit Button Overrides */
    div.stButton {
        text-align: center; /* Ensures button wrapper is centered */
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); 
        color: white; 
        border: none; 
        padding: 12px 30px; /* Wider padding */
        border-radius: 25px; 
        font-weight: 600; 
        width: 100%; /* Full width within container, constrained by column */
        max-width: 200px; /* Prevent button from becoming too wide */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    div.stButton > button:hover { 
        transform: scale(1.05); 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
    }

    /* Decorative Footer */
    .footer-deco {
        margin-top: 60px;
        height: 10px;
        background: linear-gradient(90deg, #009c3b 0%, #ffdf00 50%, #002776 100%); 
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 2. HERO SECTION
# ------------------------------------------

# Display Title
#st.markdown('<div class="main-title">THE WORLD\'S SUPPLIER</div>', unsafe_allow_html=True)
#st.markdown('<div class="subtitle">A Time Series Analysis of Brazil\'s Top Exports</div>', unsafe_allow_html=True)

# Display Hero Image
# Updated column ratios to [0.5, 8, 0.5] to give the image significantly more width
try:
    hero_img = Image.open("image_29519f.png")
    c1, c2, c3 = st.columns([0.5, 8, 0.5])
    with c2:
        st.image(hero_img, use_container_width=True)
except FileNotFoundError:
    st.error("Hero image not found. Please ensure 'image_29519f.jpg' is in the directory.")

st.markdown("---")

# ------------------------------------------
# 3. DASHBOARD NAVIGATION GRID
# ------------------------------------------

st.markdown("<h3 style='text-align: center; color: #1b5e20; margin-bottom: 40px;'>SELECT A COMMODITY DASHBOARD</h3>",
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

# 1. SOYBEAN
with col1:
    st.markdown("""
    <div class="product-card btn-soy">
        <div class="card-icon">🌱</div>
        <div class="card-title">Soybean</div>
        <p class="card-text">Global leader in production & export.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Soybean", key="btn_soy"):
        st.switch_page("pages/SoyBean.py")

# 2. CRUDE OIL
with col2:
    st.markdown("""
    <div class="product-card btn-oil">
        <div class="card-icon">🛢️</div>
        <div class="card-title">Crude Oil</div>
        <p class="card-text">Pre-salt extraction & global logistics.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Crude Oil", key="btn_oil"):
        st.switch_page("pages/CrudeOil.py")

# 3. IRON ORE
with col3:
    st.markdown("""
    <div class="product-card btn-iron">
        <div class="card-icon">🏗️</div>
        <div class="card-title">Iron Ore</div>
        <p class="card-text">Mining giants & Asian market demand.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Iron Ore", key="btn_iron"):
        st.switch_page("pages/IronOre.py")

# 4. COFFEE
with col4:
    st.markdown("""
    <div class="product-card btn-coffee">
        <div class="card-icon">☕</div>
        <div class="card-title">Coffee</div>
        <p class="card-text">Premium Arabica & Robusta flows.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Coffee", key="btn_coffee"):
        st.switch_page("pages/Coffee.py")

# 5. BOVINE MEAT
with col5:
    st.markdown("""
    <div class="product-card btn-meat">
        <div class="card-icon">🥩</div>
        <div class="card-title">Bovine Meat</div>
        <p class="card-text">Cold chain logistics & food safety.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Meat", key="btn_meat"):
        st.switch_page("pages/Meat.py")

# ------------------------------------------
# 4. FOOTER
# ------------------------------------------
st.markdown('<div class="footer-deco"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-top: 20px; color: #888;">
    <small>Data Sources: MDIC, COMEX STAT | Developed with Streamlit</small>
</div>
""", unsafe_allow_html=True)
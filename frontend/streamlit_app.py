import sys
import os

# Ensure project root is on the path regardless of where streamlit is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import date

from src.pipeline.prediction_pipeline import PredictionPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyChainIQ – Delay Risk Scorer",
    page_icon="📦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero p  { font-size: 1rem; opacity: 0.75; margin-top: 0.3rem; }

.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6c8ebf;
    margin: 1.4rem 0 0.5rem;
}

.risk-card {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    color: white;
    text-align: center;
}
.risk-high   { background: linear-gradient(135deg, #c0392b, #e74c3c); }
.risk-medium { background: linear-gradient(135deg, #e67e22, #f39c12); }
.risk-low    { background: linear-gradient(135deg, #27ae60, #2ecc71); }
.risk-card h2 { font-size: 2.8rem; font-weight: 800; margin: 0; }
.risk-card p  { font-size: 1.05rem; margin: 0.3rem 0 0; opacity: 0.9; }

.metric-tile {
    background: #1a1f2e;
    border: 1px solid #2d3545;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-tile .label { font-size: 0.78rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-tile .value { font-size: 1.9rem; font-weight: 700; color: #e0e8ff; margin-top: 0.2rem; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4776e6, #8e54e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📦 SupplyChainIQ</h1>
  <p>Shipment Delay Risk Intelligence · Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ── Load model pipeline ───────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown('<div class="section-title">🚚 Shipment & Order Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        shipping_mode    = st.selectbox("Shipping Mode", ["Standard Class", "First Class", "Second Class", "Same Day"])
        order_type       = st.selectbox("Order Type", ["DEBIT", "TRANSFER", "PAYMENT", "CASH"])
        days_scheduled   = st.slider("Days Scheduled for Shipment", 0, 4, 4)
    with c2:
        customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
        market           = st.selectbox("Market", ["Pacific Asia", "Europe", "LATAM", "USCA", "Africa"])
        order_region     = st.text_input("Order Region", "Southeast Asia")
    with c3:
        order_status = st.selectbox("Order Status", [
            "COMPLETE", "PENDING", "CLOSED", "PROCESSING",
            "PENDING_PAYMENT", "CANCELED", "ON_HOLD",
            "SUSPECTED_FRAUD", "PAYMENT_REVIEW"
        ])
        order_date = st.date_input("Order Date", date(2017, 1, 15))

    st.markdown('<div class="section-title">💰 Financials</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        discount_rate      = st.number_input("Discount Rate", value=0.05, step=0.01, format="%.2f")
        discount           = st.number_input("Order Item Discount ($)", value=15.0)
    with f2:
        product_price      = st.number_input("Product Price (Order Item) ($)", value=300.0)
        profit_ratio       = st.number_input("Profit Ratio", value=0.25, step=0.01, format="%.2f")
    with f3:
        quantity           = st.number_input("Item Quantity", value=1, min_value=1, step=1)
        sales              = st.number_input("Sales ($)", value=300.0)
    with f4:
        benefit_per_order  = st.number_input("Benefit per Order ($)", value=80.0)
        sales_per_customer = st.number_input("Sales per Customer ($)", value=300.0)

    with st.expander("📋 Category / Product / Location details"):
        e1, e2, e3 = st.columns(3)
        with e1:
            category_id         = st.number_input("Category Id", value=73, step=1)
            category_name       = st.text_input("Category Name", "Sporting Goods")
            department_id       = st.number_input("Department Id", value=2, step=1)
            department_name     = st.text_input("Department Name", "Fitness")
            product_category_id = st.number_input("Product Category Id", value=73, step=1)
        with e2:
            product_name        = st.text_input("Product Name", "Diamondback Women's Serene Classic Comfort Bike")
            product_status      = st.selectbox("Product Status", [0, 1])
            product_price_field = st.number_input("Product Price ($)", value=300.0)
        with e3:
            customer_city       = st.text_input("Customer City", "Caguas")
            customer_country    = st.text_input("Customer Country", "EE. UU.")
            customer_state      = st.text_input("Customer State", "PR")
            order_city          = st.text_input("Order City", "Bangkok")
            order_country       = st.text_input("Order Country", "Thailand")
            order_state         = st.text_input("Order State", "Bangkok")

        g1, g2 = st.columns(2)
        with g1:
            order_profit_per_order = st.number_input("Order Profit Per Order ($)", value=80.0)
        with g2:
            order_item_total       = st.number_input("Order Item Total ($)", value=300.0)

    submitted = st.form_submit_button("🔍 Assess Delivery Risk", type="primary")

# ── Prediction & Results ──────────────────────────────────────────────────────
if submitted:
    raw_row = {
        'Type':                          order_type,
        'Days for shipment (scheduled)': int(days_scheduled),
        'Benefit per order':             benefit_per_order,
        'Sales per customer':            sales_per_customer,
        'Category Id':                   int(category_id),
        'Category Name':                 category_name,
        'Customer City':                 customer_city,
        'Customer Country':              customer_country,
        'Customer Segment':              customer_segment,
        'Customer State':                customer_state,
        'Department Id':                 int(department_id),
        'Department Name':               department_name,
        'Market':                        market,
        'Order City':                    order_city,
        'Order Country':                 order_country,
        'Order Item Discount':           discount,
        'Order Item Discount Rate':      discount_rate,
        'Order Item Product Price':      product_price,
        'Order Item Profit Ratio':       profit_ratio,
        'Order Item Quantity':           int(quantity),
        'Sales':                         sales,
        'Order Item Total':              order_item_total,
        'Order Profit Per Order':        order_profit_per_order,
        'Order Region':                  order_region,
        'Order State':                   order_state,
        'Order Status':                  order_status,
        'Product Category Id':           int(product_category_id),
        'Product Name':                  product_name,
        'Product Price':                 product_price_field,
        'Product Status':                int(product_status),
        'Shipping Mode':                 shipping_mode,
        'order date (DateOrders)':       str(order_date),
    }

    with st.spinner("Scoring shipment risk…"):
        try:
            result = pipeline.predict(raw_row)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    risk_score = result['risk_score']
    delay_prob = result['delay_probability']
    prediction = result['prediction']

    if risk_score >= 70:
        risk_level, css_class, emoji = "HIGH RISK", "risk-high", "🔴"
    elif risk_score >= 40:
        risk_level, css_class, emoji = "MEDIUM RISK", "risk-medium", "🟡"
    else:
        risk_level, css_class, emoji = "LOW RISK", "risk-low", "🟢"

    st.divider()
    st.markdown("### 📊 Risk Report")

    st.markdown(f"""
    <div class="risk-card {css_class}">
      <p style="font-size:1.1rem; margin-bottom:0.4rem">{emoji} {risk_level}</p>
      <h2>{risk_score}/100</h2>
      <p>Prediction: <strong>{prediction}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="label">Risk Score</div>
          <div class="value">{risk_score}/100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="label">Delay Probability</div>
          <div class="value">{delay_prob*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="label">Prediction</div>
          <div class="value">{prediction}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Delay Probability** — `{delay_prob*100:.1f}%`")
    st.progress(float(delay_prob))

    st.caption("ℹ️ Risk factors breakdown (SHAP explainability) coming in next release.")

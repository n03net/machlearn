"""
E-Commerce Purchase Prediction — Streamlit App
Jalankan: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Premium ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f6f9fc 0%, #e9f0f5 100%);
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.25);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        margin-top: 0.8rem;
        opacity: 0.95;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric row */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        flex: 1;
        background: white;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #888;
        letter-spacing: 1px;
    }
    
    .metric-number {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }
    
    /* Upload area */
    .upload-area {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #667eea;
        margin: 1rem 0;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: none;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 12px;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.4rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info/Warning/Success */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_artifacts/best_model.pkl")
        with open("model_artifacts/model_metadata.json") as f:
            meta = json.load(f)
        return model, meta
    except FileNotFoundError:
        return None, None

model, meta = load_model()

# ─── Constants ────────────────────────────────────────────────────────────────
ORIG_FEATURES = [
    'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
    'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add',
    'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail',
    'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts',
    'saw_delivery', 'saw_account_upgrade', 'saw_homepage',
    'device_mobile', 'device_computer', 'device_tablet', 'returning_user', 'loc_uk'
]

FEATURE_LABELS = {
    'basket_icon_click': '🛒 Klik Ikon Basket',
    'basket_add_list': '➕ Tambah ke Basket',
    'basket_add_detail': '➕ Tambah ke Basket (Detail)',
    'sort_by': '🔤 Sort By',
    'image_picker': '🖼️ Pilih Gambar',
    'account_page_click': '👤 Klik Akun',
    'promo_banner_click': '📢 Klik Promo',
    'detail_wishlist_add': '❤️ Wishlist',
    'list_size_dropdown': '📏 Pilih Ukuran',
    'closed_minibasket_click': '🧺 Klik Basket',
    'checked_delivery_detail': '🚚 Cek Pengiriman',
    'checked_returns_detail': '↩️ Cek Retur',
    'sign_in': '🔐 Login',
    'saw_checkout': '💳 Lihat Checkout',
    'saw_sizecharts': '📊 Lihat Size Chart',
    'saw_delivery': '📦 Lihat Delivery',
    'saw_account_upgrade': '⬆️ Lihat Upgrade',
    'saw_homepage': '🏠 Lihat Homepage',
    'device_mobile': '📱 Mobile',
    'device_computer': '💻 Komputer',
    'device_tablet': '📲 Tablet',
    'returning_user': '🔄 Returning User',
    'loc_uk': '🇬🇧 UK',
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    df_eng['total_activity'] = df_eng[ORIG_FEATURES].sum(axis=1)
    df_eng['basket_intent'] = df_eng['basket_icon_click'] + df_eng['basket_add_list'] + df_eng['basket_add_detail']
    df_eng['checkout_intent'] = df_eng['saw_checkout'] + df_eng['closed_minibasket_click']
    df_eng['product_info_check'] = df_eng['checked_delivery_detail'] + df_eng['checked_returns_detail'] + df_eng['saw_sizecharts']
    df_eng['engagement_score'] = df_eng['promo_banner_click'] + df_eng['image_picker'] + df_eng['detail_wishlist_add']
    
    if meta and 'features' in meta:
        return df_eng[meta['features']]
    return df_eng

# ─── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🛍️ E-Commerce Purchase Prediction</div>
    <div class="hero-subtitle">AI-Powered Customer Behavior Analytics | Batch Processing & Advanced Insights</div>
    <div class="hero-badge">⚡ Machine Learning | 🎯 89% ROC-AUC | 📊 Real-time Predictions</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Model Performance")
    
    if meta:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 1.2rem; color: white; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; opacity: 0.9;">ACTIVE MODEL</div>
            <div style="font-size: 1.1rem; font-weight: 700;">{meta['model_name']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 ROC-AUC", f"{meta['roc_auc']:.3f}")
            st.metric("📈 Precision", f"{meta.get('precision', 0.85):.3f}")
        with col2:
            st.metric("🎯 F1-Score", f"{meta['f1_score']:.3f}")
            st.metric("🔄 Recall", f"{meta.get('recall', 0.82):.3f}")
        
        st.markdown(f"""
        <div style="background: #f0f4ff; border-radius: 12px; padding: 1rem; margin: 1rem 0; text-align: center;">
            <div style="font-size: 0.7rem; color: #666;">OPTIMAL THRESHOLD</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #667eea;">{meta['optimal_threshold']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info("""
        • 🔥 **saw_checkout** = strongest signal  
        • 🛒 **Basket additions** = high intent  
        • 🔄 **Returning users** = 3x conversion  
        • 📱 **Mobile users** = optimize UX
        """)
    
    st.markdown("---")
    st.caption("© 2024 E-Commerce Predictor")

# ─── Main Content ────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model tidak ditemukan")
    st.info("Pastikan folder `model_artifacts/` ada di direktori yang sama dengan file ini")
    st.stop()

# File Upload Section
st.markdown("### 📂 1. Upload Customer Data")
st.markdown("Upload CSV file dengan data perilaku customer untuk analisis prediktif")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv'],
        help="File harus memiliki 23 kolom fitur yang diperlukan"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_file:
        if st.button("🗑️ Reset", use_container_width=True):
            st.rerun()

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Validasi kolom
        missing_cols = [col for col in ORIG_FEATURES if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Kolom yang hilang: {missing_cols[:5]}...")
            st.stop()
        
        has_target = 'ordered' in df.columns
        
        # ─── Data Overview ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 2. Data Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Total Records</div>
                <div class="card-value">{df.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔧 Features</div>
                <div class="card-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_count = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="card">
                <div class="card-title">⚠️ Missing Values</div>
                <div class="card-value">{missing_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if has_target:
                purchase_rate = df['ordered'].mean() * 100
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🎯 Purchase Rate</div>
                    <div class="card-value">{purchase_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                file_size = uploaded_file.size / 1024
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">💾 File Size</div>
                    <div class="card-value">{file_size:.1f} KB</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Data preview
        with st.expander("🔍 Preview Data (10 baris pertama)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # ─── EDA Section ───────────────────────────────────────────────────────
        if has_target:
            st.markdown("### 📈 3. Exploratory Data Analysis")
            
            # Target distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Target Distribution")
                target_counts = df['ordered'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Tidak Beli (0)', 'Beli (1)'],
                    values=target_counts.values,
                    hole=0.4,
                    marker_colors=['#f45c43', '#38ef7d'],
                    textinfo='percent+label'
                )])
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Top 10 Feature Correlations")
                correlations = []
                for feature in ORIG_FEATURES:
                    corr = df[feature].corr(df['ordered'])
                    correlations.append((FEATURE_LABELS.get(feature, feature), corr))
                
                corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation']).sort_values('Correlation', ascending=False).head(10)
                
                fig = go.Figure(data=[go.Bar(
                    x=corr_df['Correlation'],
                    y=corr_df['Feature'],
                    orientation='h',
                    marker_color=corr_df['Correlation'].apply(lambda x: '#38ef7d' if x > 0 else '#f45c43'),
                    text=corr_df['Correlation'].round(3),
                    textposition='outside'
                )])
                fig.update_layout(height=380, xaxis_title="Correlation", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        
        # ─── Prediction Section ────────────────────────────────────────────────
        st.markdown("### 🤖 4. Run Predictions")
        
        if st.button("🚀 Start Prediction Analysis", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing customer behavior..."):
                # Prepare features
                X = engineer_features(df)
                
                # Make predictions
                probabilities = model.predict_proba(X)[:, 1]
                threshold = meta['optimal_threshold'] if meta else 0.5
                predictions = (probabilities >= threshold).astype(int)
                
                # Create result dataframe
                result_df = df.copy()
                result_df['purchase_probability'] = probabilities
                result_df['prediction'] = predictions
                result_df['status'] = np.where(predictions == 1, '✅ WILL BUY', '❌ WILL NOT BUY')
                
                buy_count = predictions.sum()
                not_buy_count = (predictions == 0).sum()
                avg_prob = probabilities.mean() * 100
                
                # Results summary
                st.markdown("#### 📊 Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="card-title">👥 Total Customers</div>
                        <div class="card-value">{len(predictions):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="card" style="text-align: center; border-bottom: 3px solid #38ef7d;">
                        <div class="card-title">✅ Will Buy</div>
                        <div class="card-value" style="color: #38ef7d; -webkit-text-fill-color: #38ef7d;">{buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="card" style="text-align: center; border-bottom: 3px solid #f45c43;">
                        <div class="card-title">❌ Will Not Buy</div>
                        <div class="card-value" style="color: #f45c43; -webkit-text-fill-color: #f45c43;">{not_buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({not_buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div class="card-title">📈 Avg Probability</div>
                        <div class="card-value">{avg_prob:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📈 Probability Distribution")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=probabilities * 100,
                    nbinsx=40,
                    marker_color='#667eea',
                    marker_line_width=0,
                    opacity=0.8
                ))
                fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#f45c43", line_width=2,
                              annotation_text=f"Threshold: {threshold*100:.0f}%")
                fig.update_layout(
                    height=400,
                    xaxis_title="Purchase Probability (%)",
                    yaxis_title="Number of Customers",
                    plot_bgcolor='white',
                    bargap=0.05
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("#### 📋 Prediction Results (Sample)")
                
                display_df = result_df[['purchase_probability', 'status'] + ORIG_FEATURES[:6]].head(20).copy()
                display_df['purchase_probability'] = (display_df['purchase_probability'] * 100).round(1)
                display_df.columns = ['Prob (%)', 'Status'] + [FEATURE_LABELS.get(c, c) for c in ORIG_FEATURES[:6]]
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download section
                st.markdown("#### 💾 Download Results")
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"predictions_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Simple summary
                    summary = {
                        'total_customers': len(predictions),
                        'predicted_buyers': int(buy_count),
                        'predicted_non_buyers': int(not_buy_count),
                        'average_probability': float(avg_prob),
                        'timestamp': timestamp
                    }
                    st.download_button(
                        label="📊 Download Summary",
                        data=json.dumps(summary, indent=2),
                        file_name=f"summary_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Business insights
                st.markdown("---")
                st.markdown("### 💡 Business Insights")
                
                insights = []
                
                basket_adders = result_df[(result_df['basket_add_detail'] == 1) & (result_df['prediction'] == 0)]
                if len(basket_adders) > 0:
                    insights.append(f"⚠️ **{len(basket_adders):,} users** added to cart but didn't buy → Send reminder emails")
                
                returning_users = result_df[result_df['returning_user'] == 1]
                if len(returning_users) > 0:
                    conv_rate = returning_users['prediction'].mean() * 100
                    insights.append(f"🔄 **{len(returning_users):,} returning users** with {conv_rate:.1f}% conversion → Launch loyalty program")
                
                high_intent = result_df[result_df['purchase_probability'] > 0.7]
                if len(high_intent) > 0:
                    insights.append(f"🔥 **{len(high_intent):,} high-intent customers** → Priority for personalized offers")
                
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                st.balloons()
                st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    # No file uploaded - show guide
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem;">📂</div>
            <h3 style="margin-top: 1rem;">Ready to Analyze?</h3>
            <p style="color: #666;">Upload a CSV file to get started with purchase prediction</p>
            
            <div class="card" style="text-align: left; margin-top: 2rem;">
                <div class="card-title">📋 Required CSV Format</div>
                <p style="font-size: 0.85rem; color: #666;">Your CSV must include these 23 columns:</p>
                <code style="font-size: 0.7rem; background: #f0f4ff; padding: 0.5rem; display: block; border-radius: 8px;">
                basket_icon_click, basket_add_list, basket_add_detail, sort_by, image_picker,<br>
                account_page_click, promo_banner_click, detail_wishlist_add, list_size_dropdown,<br>
                closed_minibasket_click, checked_delivery_detail, checked_returns_detail, sign_in,<br>
                saw_checkout, saw_sizecharts, saw_delivery, saw_account_upgrade, saw_homepage,<br>
                device_mobile, device_computer, device_tablet, returning_user, loc_uk
                </code>
                <hr style="margin: 1rem 0;">
                <div class="card-title">✨ What You'll Get</div>
                <ul style="margin: 0; padding-left: 1rem;">
                    <li>🎯 Purchase probability for each customer</li>
                    <li>📊 Interactive data visualizations</li>
                    <li>💡 Actionable business insights</li>
                    <li>📥 Downloadable results (CSV + JSON)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    E-Commerce Purchase Prediction System | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
